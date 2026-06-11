"""Unit tests for PipelineStage MicrobatchKey support.

Run:
    PYTHONPATH=. python femtotron/test/unit/test_pipeline_stage_task_key.py
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from femtotron.parallel.pipeline_parallel.scheduler.ir import TaskKey
from femtotron.parallel.pipeline_parallel.stage import MicrobatchKey, PipelineStage


class DummyParallelContext:
    pp_size = 1
    pp_rank = 0


class ToyMiddleModel(nn.Module):
    is_first = False
    is_last = False

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return {"hidden_states": self.linear(x)}


class ToyLastModel(nn.Module):
    is_first = False
    is_last = True

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        logits = self.linear(x)
        if labels is None:
            return {"logits": logits}
        loss = torch.nn.functional.mse_loss(logits, labels)
        return {"loss": loss}


def log(msg: str) -> None:
    print(f"  {msg}")


def assert_raises(
    expected_type: type[Exception],
    fn: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> Exception:
    try:
        fn(*args, **kwargs)
    except expected_type as exc:
        return exc
    raise AssertionError(f"expected {expected_type.__name__} from {fn.__name__}")


def make_middle_stage(seed: int = 0) -> PipelineStage:
    torch.manual_seed(seed)
    return PipelineStage(ToyMiddleModel(), DummyParallelContext())


def run_middle_backward(
    stage: PipelineStage,
    mb: MicrobatchKey,
    x: torch.Tensor,
) -> torch.Tensor:
    stage.stage_input(mb, x)
    stage.forward(mb)
    out = stage.get_output(mb)
    assert out.shape == x.shape
    stage.stage_grad(mb, torch.randn_like(out))
    stage.backward(mb)
    return stage.get_input_grad(mb)


def test_pipeline_stage_still_accepts_int_key() -> None:
    stage = make_middle_stage()
    x = torch.randn(2, 4)

    input_grad = run_middle_backward(stage, 0, x)

    assert input_grad.shape == x.shape
    stage.assert_clean()
    log("PipelineStage still accepts legacy int mb_id keys")


def test_pipeline_stage_accepts_task_key() -> None:
    stage = make_middle_stage()
    mb = TaskKey(mb=0, logical_stage=1, local_stage=0)
    x = torch.randn(2, 4)

    input_grad = run_middle_backward(stage, mb, x)

    assert input_grad.shape == x.shape
    stage.assert_clean()
    log("PipelineStage accepts TaskKey microbatch identities")


def test_task_keys_with_same_mb_do_not_collide() -> None:
    stage = make_middle_stage()
    mb0 = TaskKey(mb=0, logical_stage=1, local_stage=0)
    mb1 = TaskKey(mb=0, logical_stage=5, local_stage=1)
    x0 = torch.randn(2, 4)
    x1 = torch.randn(2, 4)

    stage.stage_input(mb0, x0)
    stage.stage_input(mb1, x1)
    stage.forward(mb0)
    stage.forward(mb1)
    out0 = stage.get_output(mb0)
    out1 = stage.get_output(mb1)
    stage.stage_grad(mb0, torch.randn_like(out0))
    stage.stage_grad(mb1, torch.randn_like(out1))
    stage.backward(mb0)
    stage.backward(mb1)

    grad0 = stage.get_input_grad(mb0)
    grad1 = stage.get_input_grad(mb1)
    assert grad0.shape == x0.shape
    assert grad1.shape == x1.shape
    stage.assert_clean()
    log("TaskKey microbatch identities with the same mb do not collide")


def test_duplicate_stage_input_with_task_key_raises() -> None:
    stage = make_middle_stage()
    mb = TaskKey(mb=0, logical_stage=1)

    stage.stage_input(mb, torch.randn(2, 4))
    exc = assert_raises(RuntimeError, stage.stage_input, mb, torch.randn(2, 4))

    assert "input already staged" in str(exc)
    stage.reset()
    stage.assert_clean()
    log("Duplicate TaskKey input staging still raises")


def test_assert_clean_formats_task_key_and_mixed_keys() -> None:
    stage = make_middle_stage()
    task_key = TaskKey(mb=0, logical_stage=3, local_stage=1)

    stage.stage_input(0, torch.randn(2, 4))
    stage.stage_input(task_key, torch.randn(2, 4))
    exc = assert_raises(RuntimeError, stage.assert_clean)

    msg = str(exc)
    assert "_inputs" in msg
    assert "0" in msg
    assert "TaskKey(" in msg
    assert "mb=0,ls=3,local=1,phase=0,dir=forward" in msg
    stage.reset()
    stage.assert_clean()
    log("assert_clean handles mixed int and TaskKey keys")


def test_backward_split_methods_accept_task_key() -> None:
    stage = make_middle_stage()
    mb = TaskKey(mb=0, logical_stage=1)
    x = torch.randn(2, 4)

    stage.stage_input(mb, x)
    stage.forward(mb)
    out = stage.get_output(mb)
    stage.stage_grad(mb, torch.randn_like(out))

    stage.backward_input_grad(mb)
    assert all(p.grad is None for p in stage.model.parameters())
    input_grad = stage.get_input_grad(mb)
    assert input_grad.shape == x.shape

    stage.backward_weight_grad(mb)
    assert any(p.grad is not None for p in stage.model.parameters())
    stage.assert_clean()
    log("D/W split methods accept TaskKey keys")


def test_last_stage_loss_values_can_use_task_key() -> None:
    torch.manual_seed(0)
    stage = PipelineStage(ToyLastModel(), DummyParallelContext())
    mb = TaskKey(mb=0, logical_stage=3)
    x = torch.randn(2, 4)
    labels = torch.randn(2, 4)

    stage.stage_input(mb, x)
    stage.stage_labels(mb, labels)
    stage.forward(mb)
    stage.backward(mb)
    input_grad = stage.get_input_grad(mb)
    losses = stage.pop_all_losses()

    assert input_grad.shape == x.shape
    assert set(losses.keys()) == {mb}
    assert losses[mb].ndim == 0
    stage.assert_clean()
    log("Last-stage losses can be keyed by TaskKey")


def main() -> None:
    tests = [
        test_pipeline_stage_still_accepts_int_key,
        test_pipeline_stage_accepts_task_key,
        test_task_keys_with_same_mb_do_not_collide,
        test_duplicate_stage_input_with_task_key_raises,
        test_assert_clean_formats_task_key_and_mixed_keys,
        test_backward_split_methods_accept_task_key,
        test_last_stage_loss_values_can_use_task_key,
    ]
    print(f"\nRunning {len(tests)} PipelineStage MicrobatchKey tests\n")
    for test in tests:
        print(f"[{test.__name__}]")
        test()
    print(f"\nAll {len(tests)} PipelineStage MicrobatchKey tests passed\n")


if __name__ == "__main__":
    main()
