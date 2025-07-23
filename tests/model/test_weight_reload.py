"""Model weights survive a checkpoint round-trip exactly (torch extra)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.model


def test_loaded_model_reproduces_forward(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from wavrail.model.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
    from wavrail.model.network import AcousticModel

    torch.manual_seed(0)
    model = AcousticModel(input_dim=5, hidden_dim=8, vocab_size=4)
    torch.manual_seed(7)
    x = torch.randn(1, 4, 5)
    with torch.no_grad():
        expected = model(x)

    path = tmp_path / "weights.pt"
    save_checkpoint(Checkpoint(epoch=0, config={}, model_state=model.state_dict()), path)
    restored = load_checkpoint(path)

    torch.manual_seed(123)  # different init on purpose
    fresh = AcousticModel(input_dim=5, hidden_dim=8, vocab_size=4)
    fresh.load_state_dict(restored.model_state)
    with torch.no_grad():
        actual = fresh(x)
    assert torch.allclose(expected, actual)


def test_fresh_models_differ_before_reload() -> None:
    torch = pytest.importorskip("torch")
    from wavrail.model.network import AcousticModel

    torch.manual_seed(0)
    a = AcousticModel(input_dim=5, hidden_dim=8, vocab_size=4)
    torch.manual_seed(123)
    b = AcousticModel(input_dim=5, hidden_dim=8, vocab_size=4)
    torch.manual_seed(7)
    x = torch.randn(1, 4, 5)
    with torch.no_grad():
        assert not torch.allclose(a(x), b(x))
