"""End-to-end train → decode → score example (requires the torch extra)."""

from __future__ import annotations

import json

from wavrail.model import (
    TrainingConfig,
    build_synthetic_dataset,
    train_synthetic,
)
from wavrail.vocab import Vocab


def main() -> int:
    vocab = Vocab.from_labels(["<blank>", "a", "b", "c", "d"])

    def factory(seed: int) -> tuple[list, list]:
        examples = build_synthetic_dataset(vocab, n_examples=8, duration_s=0.25, seed=seed)
        return examples[:6], examples[6:]

    config = TrainingConfig(n_epochs=4, batch_size=2, learning_rate=5e-3, hidden_dim=16)
    report = train_synthetic(config, factory)
    train_examples, _dev = factory(0)
    reference = train_examples[0].text
    payload = {
        "initial_loss": report.initial_loss,
        "final_loss": report.final_loss,
        "loss_decreased": report.loss_decreased(),
        "reference_example": reference,
        "dev_cer": report.dev_cer,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
