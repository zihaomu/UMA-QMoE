from __future__ import annotations

from types import SimpleNamespace

import pytest

from uma_qmoe.experiments.quantizers import QuantizedCandidate
from uma_qmoe.experiments.session import ProjectionTransaction, QwenProjectionSession
from uma_qmoe.experiments.types import Unit


class _ShapeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.ndim = len(shape)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            shape = []
            for size, selector in zip(self.shape, key, strict=True):
                if isinstance(selector, int):
                    continue
                if isinstance(selector, slice):
                    start, stop, step = selector.indices(size)
                    shape.append(max(0, (stop - start + step - 1) // step))
            return _ShapeTensor(tuple(shape))
        return _ShapeTensor(self.shape[1:])


def test_qwen_session_discovers_exactly_72_projection_units() -> None:
    experts = SimpleNamespace(
        gate_up_proj=_ShapeTensor((60, 2816, 2048)),
        down_proj=_ShapeTensor((60, 2048, 1408)),
    )
    layer = SimpleNamespace(mlp=SimpleNamespace(experts=experts))
    model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="qwen2_moe",
            num_hidden_layers=24,
            num_experts=60,
            num_experts_per_tok=4,
        ),
        model=SimpleNamespace(layers=[layer] * 24),
    )

    session = QwenProjectionSession(model, torch_module=SimpleNamespace())

    assert len(session.discover_units()) == 72
    assert session.discover_units()[0] == Unit(0, "gate")
    assert session.discover_units()[-1] == Unit(23, "down")


def test_projection_transaction_restores_after_error() -> None:
    torch = pytest.importorskip("torch")
    target = torch.arange(128, dtype=torch.float32)
    snapshot = target.clone()
    session = SimpleNamespace(torch=torch, _active=Unit(0, "gate"))
    transaction = ProjectionTransaction(
        session=session,
        unit=Unit(0, "gate"),
        target=target,
        snapshot=snapshot,
        original_summary={
            "shape": [128],
            "dtype": str(snapshot.dtype),
            "device": str(snapshot.device),
            "element_count": 128,
            "sum": float(snapshot.float().sum().item()),
            "sum_abs": float(snapshot.float().abs().sum().item()),
        },
    )

    with pytest.raises(RuntimeError, match="trial failed"):
        with transaction:
            transaction.apply(QuantizedCandidate(restored_weight=torch.zeros_like(target)))
            assert torch.count_nonzero(target) == 0
            raise RuntimeError("trial failed")

    assert torch.equal(target, snapshot)
    assert transaction.restoration is not None
    assert transaction.restoration["exact"] is True
    assert session._active is None
