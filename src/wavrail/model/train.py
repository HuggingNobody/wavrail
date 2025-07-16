"""Training entry point for the synthetic acoustic model.

The trainer is intentionally small and CPU-friendly so it can
serve as a worked example without GPU access or pretrained
weights. All randomness is seeded so the resulting loss curve
is reproducible.

The trainer records the per-epoch loss and, when possible, the
held-out synthetic CER. Any real evaluation claims live in docs
and READMEs that explicitly cite a synthetic demo, never a real
benchmark.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from wavrail.errors import ModelError
from wavrail.vocab import Vocab


@dataclass
class TrainingConfig:
    """Hyperparameters for :func:`train_synthetic`."""

    n_epochs: int = 4
    batch_size: int = 2
    learning_rate: float = 1e-3
    grad_clip: float = 1.0
    hidden_dim: int = 32
    seed: int = 0
    lr_decay_factor: float = 1.0
    lr_decay_every: int = 0

    def __post_init__(self) -> None:
        if self.n_epochs < 0:
            raise ModelError("n_epochs must be >= 0")
        if self.batch_size < 1:
            raise ModelError("batch_size must be >= 1")
        if self.learning_rate <= 0:
            raise ModelError("learning_rate must be > 0")
        if self.grad_clip <= 0:
            raise ModelError("grad_clip must be > 0")
        if self.hidden_dim < 1:
            raise ModelError("hidden_dim must be >= 1")
        if self.lr_decay_factor <= 0:
            raise ModelError("lr_decay_factor must be > 0")
        if self.lr_decay_every < 0:
            raise ModelError("lr_decay_every must be >= 0")


@dataclass
class TrainingReport:
    """Result of a training run."""

    epoch_losses: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    resumed_from_epoch: int = 0
    final_loss: float = 0.0
    initial_loss: float = 0.0
    n_train_examples: int = 0
    n_dev_examples: int = 0
    dev_cer: float | None = None

    def loss_decreased(self) -> bool:
        """Return True if the final loss is strictly less than the initial."""
        return self.final_loss < self.initial_loss


def _require_torch():
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ModelError(
            "the optional 'torch' extra is required for acoustic-model training; "
            "install with `pip install wavrail[torch]`"
        ) from exc
    return torch


def _lr_for_epoch(config: TrainingConfig, epoch: int) -> float:
    """Step-decay schedule: ``lr * factor ** (epoch // every)``.

    ``lr_decay_every <= 0`` disables the schedule and keeps the
    configured learning rate for every epoch.
    """
    if config.lr_decay_every <= 0:
        return config.learning_rate
    return config.learning_rate * config.lr_decay_factor ** (epoch // config.lr_decay_every)


def train_synthetic(
    config: TrainingConfig,
    dataset_factory: Callable[[int], tuple[list, list]],
    *,
    vocab: Vocab | None = None,
    resume_from: str | Path | None = None,
    save_to: str | Path | None = None,
) -> TrainingReport:
    """Train a small CTC acoustic model on a synthetic dataset.

    ``dataset_factory`` is invoked with a seed and returns a
    ``(train, dev)`` pair of :class:`wavrail.model.SyntheticExample`
    lists. The training loop minimises the CTC negative
    log-likelihood (``torch.nn.functional.ctc_loss``) with gradient
    clipping; all randomness is seeded so the loss curve is
    reproducible.

    When ``vocab`` is supplied it sizes the CTC head and labels
    decoded ids during dev evaluation, so ``dev_cer`` is directly
    comparable to the public character-level metric. The dev split
    is tiny synthetic formant audio: a small model routinely fails
    to generalise to it, and a dev CER at or above 1.0 is reported
    as-is rather than hidden. It is a pipeline-health signal, not
    an accuracy claim.
    """
    torch = _require_torch()
    train_examples, dev_examples = dataset_factory(config.seed)
    report = TrainingReport(n_train_examples=len(train_examples), n_dev_examples=len(dev_examples))
    if not train_examples:
        raise ModelError("training dataset is empty")
    from wavrail.model.network import AcousticModel, build_targets

    torch.manual_seed(config.seed)
    # without a dataset vocab, fall back to a head width that comfortably
    # exceeds any target id the synthetic builder can emit
    fallback_width = train_examples[0].features.shape[1] + 8
    model_vocab_size = vocab.size if vocab is not None else fallback_width
    model = AcousticModel(
        input_dim=train_examples[0].features.shape[1],
        hidden_dim=config.hidden_dim,
        vocab_size=model_vocab_size,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    start_epoch = 0
    if resume_from is not None:
        start_epoch = _apply_resume(model, optimiser, config, report, resume_from)
    for epoch in range(start_epoch, config.n_epochs):
        lr = _lr_for_epoch(config, epoch)
        for group in optimiser.param_groups:
            group["lr"] = lr
        report.learning_rates.append(lr)
        total_loss = 0.0
        n_batches = 0
        for batch_start in range(0, len(train_examples), config.batch_size):
            batch = train_examples[batch_start : batch_start + config.batch_size]
            inputs = torch.tensor(
                np.array([example.features for example in batch]), dtype=torch.float32
            )
            targets, target_lengths = build_targets(batch)
            logits = model(inputs)
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = _ctc_loss(log_probs, targets, target_lengths)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimiser.step()
            total_loss += float(loss.item())
            n_batches += 1
        epoch_loss = total_loss / max(n_batches, 1)
        if epoch == 0:
            report.initial_loss = epoch_loss
        report.epoch_losses.append(epoch_loss)
    report.final_loss = report.epoch_losses[-1] if report.epoch_losses else 0.0
    if save_to is not None:
        from wavrail.model.checkpoint import Checkpoint, save_checkpoint

        save_checkpoint(
            Checkpoint(
                epoch=config.n_epochs,
                config=asdict(config),
                model_state=model.state_dict(),
                epoch_losses=list(report.epoch_losses),
                optimizer_state=optimiser.state_dict(),
            ),
            save_to,
        )
    if dev_examples:
        report.dev_cer = _evaluate_dev(model, dev_examples, vocab)
    return report


def _apply_resume(
    model: object,
    optimiser: object,
    config: TrainingConfig,
    report: TrainingReport,
    resume_from: str | Path,
) -> int:
    """Restore weights, optimiser state, and history; return the next epoch."""
    from wavrail.model.checkpoint import load_checkpoint

    checkpoint = load_checkpoint(resume_from)
    _check_resume_compatible(checkpoint.config, config)
    model.load_state_dict(checkpoint.model_state)  # type: ignore[attr-defined]
    if checkpoint.optimizer_state is not None:
        optimiser.load_state_dict(checkpoint.optimizer_state)  # type: ignore[attr-defined]
    report.epoch_losses.extend(checkpoint.epoch_losses)
    if checkpoint.epoch_losses:
        report.initial_loss = checkpoint.epoch_losses[0]
    report.resumed_from_epoch = checkpoint.epoch
    return checkpoint.epoch


def _check_resume_compatible(saved: dict[str, object], config: TrainingConfig) -> None:
    """Every config key except ``n_epochs`` must match the checkpoint."""
    current = asdict(config)
    for key, value in saved.items():
        if key == "n_epochs":
            continue  # extending the schedule is the point of resuming
        if key in current and current[key] != value:
            raise ModelError(
                f"checkpoint config mismatch: {key}={value!r} != current {current[key]!r}"
            )


def _ctc_loss(log_probs: object, targets: object, target_lengths: object) -> object:
    torch = _require_torch()
    batch_size, time_steps, _vocab_size = log_probs.shape  # type: ignore[attr-defined]
    input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long)
    return torch.nn.functional.ctc_loss(  # type: ignore[attr-defined]
        log_probs=log_probs.permute(1, 0, 2),  # type: ignore[attr-defined]
        targets=targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        blank=0,
        reduction="mean",
        zero_infinity=True,
    )


def _evaluate_dev(model: object, dev_examples: list, vocab: Vocab | None) -> float:
    """Held-out CER on the synthetic dev split.

    Uses the public character-level edit alignment so the number is
    directly comparable to corpus CER. When ``vocab`` is ``None``
    (or its width does not match the model head) a placeholder
    label set is derived from the model width; the rate is then
    only a pipeline-health signal because decoded ids cannot be
    mapped back to dataset tokens.
    """
    torch = _require_torch()
    from wavrail.ctc import greedy_decode
    from wavrail.metrics.cer import character_edit_ops

    vocab_size = int(model.vocab_size)  # type: ignore[attr-defined]
    if vocab is None or vocab.size != vocab_size:
        labels = ["<blank>"] + [f"sym{i}" for i in range(1, vocab_size)]
        vocab = Vocab.from_labels(labels)
    total_errors = 0
    total_length = 0
    model.eval()  # type: ignore[attr-defined]
    with torch.no_grad():  # type: ignore[attr-defined]
        for example in dev_examples:
            inputs = torch.tensor(example.features, dtype=torch.float32).unsqueeze(0)
            logits = model(inputs)
            log_probs = torch.log_softmax(logits, dim=-1)
            posteriors = log_probs[0].detach().numpy()
            hyp = greedy_decode(posteriors, vocab)
            hyp_text = "".join(vocab.id_to_token(t) for t in hyp.tokens)
            ops = character_edit_ops(example.text, hyp_text)
            total_errors += ops.errors
            total_length += ops.length
    if total_length == 0:
        return 0.0
    return total_errors / total_length


__all__ = ["TrainingConfig", "TrainingReport", "train_synthetic"]
