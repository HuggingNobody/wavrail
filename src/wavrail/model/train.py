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
from dataclasses import dataclass, field

from wavrail.errors import ModelError


@dataclass
class TrainingConfig:
    """Hyperparameters for :func:`train_synthetic`."""

    n_epochs: int = 4
    batch_size: int = 2
    learning_rate: float = 1e-3
    grad_clip: float = 1.0
    hidden_dim: int = 32
    seed: int = 0


@dataclass
class TrainingReport:
    """Result of a training run."""

    epoch_losses: list[float] = field(default_factory=list)
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


def train_synthetic(
    config: TrainingConfig,
    dataset_factory: Callable[[int], tuple[list, list]],
) -> TrainingReport:
    """Train a small CTC acoustic model on a synthetic dataset.

    ``dataset_factory`` is invoked with a seed and returns a
    ``(train, dev)`` pair of :class:`wavrail.model.SyntheticExample`
    lists. The training loop uses cross-entropy on the CTC log
    posterior matrix averaged across the time axis; this is a
    compact stand-in for a full CTC forward-backward and is
    sufficient to demonstrate that the loss decreases on the
    synthetic task.
    """
    torch = _require_torch()
    train_examples, dev_examples = dataset_factory(config.seed)
    report = TrainingReport(n_train_examples=len(train_examples), n_dev_examples=len(dev_examples))
    if not train_examples:
        raise ModelError("training dataset is empty")
    from wavrail.model.network import AcousticModel, build_targets

    torch.manual_seed(config.seed)
    model = AcousticModel(
        input_dim=train_examples[0].features.shape[1],
        hidden_dim=config.hidden_dim,
        vocab_size=train_examples[0].features.shape[1] + 8,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for epoch in range(config.n_epochs):
        total_loss = 0.0
        n_batches = 0
        for batch_start in range(0, len(train_examples), config.batch_size):
            batch = train_examples[batch_start : batch_start + config.batch_size]
            inputs = torch.tensor([example.features for example in batch], dtype=torch.float32)
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
    if dev_examples:
        report.dev_cer = _evaluate_dev(model, dev_examples)
    return report


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


def _evaluate_dev(model: object, dev_examples: list) -> float:
    torch = _require_torch()
    from wavrail.ctc import greedy_decode
    from wavrail.vocab import Vocab

    vocab_size = int(model.vocab_size)  # type: ignore[attr-defined]
    labels = ["<blank>"] + [f"sym{i}" for i in range(1, vocab_size)]
    vocab = Vocab.from_labels(labels)
    total_edits = 0
    total_ref = 0
    model.eval()  # type: ignore[attr-defined]
    with torch.no_grad():  # type: ignore[attr-defined]
        for example in dev_examples:
            inputs = torch.tensor(example.features, dtype=torch.float32).unsqueeze(0)
            logits = model(inputs)
            log_probs = torch.log_softmax(logits, dim=-1)
            posteriors = log_probs[0].detach().numpy()
            hyp = greedy_decode(posteriors, vocab)
            ref = list(example.text)
            hyp_chars = [vocab.id_to_token(t) for t in hyp.tokens]
            for ref_tok, hyp_tok in zip(ref, hyp_chars, strict=False):
                if ref_tok != hyp_tok:
                    total_edits += 1
            total_edits += abs(len(ref) - len(hyp_chars))
            total_ref += len(ref)
    return total_edits / max(total_ref, 1)


__all__ = ["TrainingConfig", "TrainingReport", "train_synthetic"]
