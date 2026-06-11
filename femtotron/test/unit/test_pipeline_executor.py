"""Unit tests for scheduler-v2 PipelineExecutor.

Run:
    PYTHONPATH=. python femtotron/test/unit/test_pipeline_executor.py
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from femtotron.parallel.pipeline_parallel.executor import PipelineExecutor
from femtotron.parallel.pipeline_parallel.scheduler.ir import (
    BackwardInputGradV2,
    BackwardV2,
    BackwardWeightGradV2,
    ForwardV2,
    OverlapForwardBackwardV2,
    RecvBackwardV2,
    RecvForwardV2,
    SendBackwardRecvForwardV2,
    SendBackwardV2,
    SendForwardRecvBackwardV2,
    SendForwardV2,
    TaskKey,
    WaitCommV2,
)
from femtotron.parallel.pipeline_parallel.scheduler.mapping import (
    LinearStageMapping,
    RoundRobinStageMapping,
)
from femtotron.parallel.pipeline_parallel.stage import PipelineStage


class DummyParallelContext:
    pp_size = 1
    pp_rank = 0
    world_rank = 0


class FakeComm:
    def __init__(self) -> None:
        self.sent_forward: list[torch.Tensor] = []
        self.sent_backward: list[torch.Tensor] = []
        self.forward_recvs: list[torch.Tensor] = []
        self.backward_recvs: list[torch.Tensor] = []
        self.combined_calls: list[tuple[str, torch.Tensor]] = []

    def send_forward(self, act: torch.Tensor) -> None:
        self.sent_forward.append(act)

    def recv_forward(self, out: torch.Tensor | None = None) -> torch.Tensor | None:
        assert out is not None
        out.copy_(torch.full_like(out, 2.0))
        self.forward_recvs.append(out.clone())
        return out

    def send_backward(self, grad: torch.Tensor) -> None:
        self.sent_backward.append(grad)

    def recv_backward(self, out: torch.Tensor | None = None) -> torch.Tensor | None:
        assert out is not None
        out.copy_(torch.ones_like(out))
        self.backward_recvs.append(out.clone())
        return out

    def send_forward_recv_backward(
        self,
        act: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        assert out is not None
        self.combined_calls.append(("sfrb", act))
        out.copy_(torch.ones_like(out))
        return out

    def send_backward_recv_forward(
        self,
        grad: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        assert out is not None
        self.combined_calls.append(("sbrf", grad))
        out.copy_(torch.full_like(out, 3.0))
        return out


class ToyFirstStage(nn.Module):
    is_first = True
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


class ToyMiddleStage(nn.Module):
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


class ToyLastStage(nn.Module):
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
        return {"loss": torch.nn.functional.mse_loss(logits, labels)}


class ToySingleStage(nn.Module):
    is_first = True
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
        return {"loss": torch.nn.functional.mse_loss(logits, labels)}


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


def make_stage(model: nn.Module, seed: int = 0) -> PipelineStage:
    torch.manual_seed(seed)
    return PipelineStage(model, DummyParallelContext())


def make_executor(
    stages: list[PipelineStage],
    mapping: LinearStageMapping | RoundRobinStageMapping,
    *,
    pp_rank: int = 0,
    comm: FakeComm | None = None,
) -> PipelineExecutor:
    return PipelineExecutor(
        stages=stages,
        comm=comm or FakeComm(),
        mapping=mapping,
        pp_rank=pp_rank,
        num_microbatches=2,
        recv_shape=(2, 4),
        recv_dtype=torch.float32,
    )


def test_executor_rejects_stage_count_mismatch() -> None:
    mapping = RoundRobinStageMapping(pp_size=1, virtual_stages=2)
    stage = make_stage(ToyFirstStage())

    exc = assert_raises(
        ValueError,
        PipelineExecutor,
        stages=[stage],
        comm=FakeComm(),
        mapping=mapping,
        pp_rank=0,
        num_microbatches=1,
        recv_shape=(2, 4),
        recv_dtype=torch.float32,
    )

    assert "number of local stages" in str(exc)
    log("Executor validates local stage count against mapping")


def test_executor_rejects_non_local_task_key() -> None:
    executor = make_executor(
        [make_stage(ToyFirstStage())],
        LinearStageMapping(pp_size=2),
        pp_rank=0,
    )
    non_local_key = TaskKey(mb=0, logical_stage=1, local_stage=0)

    exc = assert_raises(RuntimeError, executor.stage_for, non_local_key)

    assert "belongs to physical rank" in str(exc)
    log("Executor rejects TaskKey owned by another physical rank")


def test_executor_rejects_wrong_local_stage_in_key() -> None:
    executor = make_executor(
        [make_stage(ToyFirstStage()), make_stage(ToyLastStage())],
        RoundRobinStageMapping(pp_size=1, virtual_stages=2),
        pp_rank=0,
    )
    wrong_key = TaskKey(mb=0, logical_stage=1, local_stage=0)

    exc = assert_raises(RuntimeError, executor.stage_for, wrong_key)

    assert "mapping expects local_stage=1" in str(exc)
    log("Executor rejects TaskKey with stale local_stage")


def test_executor_runs_single_stage_forward_backward() -> None:
    executor = make_executor(
        [make_stage(ToySingleStage())],
        LinearStageMapping(pp_size=1),
    )
    key = TaskKey(mb=0, logical_stage=0, local_stage=0)

    losses = executor.run(
        [ForwardV2(key), BackwardV2(key)],
        microbatch_inputs={0: torch.randn(2, 4)},
        microbatch_labels={0: torch.randn(2, 4)},
    )

    assert set(losses.keys()) == {key}
    assert losses[key].ndim == 0
    log("Executor runs single-stage forward/backward with int payload lookup")


def test_executor_local_forward_backward_handoff() -> None:
    mapping = RoundRobinStageMapping(pp_size=1, virtual_stages=2)
    comm = FakeComm()
    executor = make_executor(
        [make_stage(ToyFirstStage()), make_stage(ToyLastStage())],
        mapping,
        pp_rank=0,
        comm=comm,
    )
    key0 = TaskKey(mb=0, logical_stage=0, local_stage=0)
    key1 = TaskKey(mb=0, logical_stage=1, local_stage=1)

    losses = executor.run(
        [
            ForwardV2(key0),
            SendForwardV2(key0),
            RecvForwardV2(key1),
            ForwardV2(key1),
            BackwardV2(key1),
            SendBackwardV2(key1),
            RecvBackwardV2(key0),
            BackwardV2(key0),
        ],
        microbatch_inputs={key0: torch.randn(2, 4)},
        microbatch_labels={key1: torch.randn(2, 4)},
    )

    assert set(losses.keys()) == {key1}
    assert not comm.sent_forward
    assert not comm.sent_backward
    assert not comm.forward_recvs
    assert not comm.backward_recvs
    log("Executor performs local forward/backward handoff without comm")


def test_executor_remote_send_forward_uses_comm() -> None:
    comm = FakeComm()
    stage = make_stage(ToyFirstStage())
    executor = make_executor(
        [stage],
        LinearStageMapping(pp_size=2),
        pp_rank=0,
        comm=comm,
    )
    key = TaskKey(mb=0, logical_stage=0, local_stage=0)

    executor.dispatch(ForwardV2(key), microbatch_inputs={0: torch.randn(2, 4)})
    executor.dispatch(SendForwardV2(key))

    assert len(comm.sent_forward) == 1
    stage.reset()
    log("Remote SendForwardV2 uses PipelineCommLike")


def test_executor_remote_recv_forward_stages_input() -> None:
    comm = FakeComm()
    stage = make_stage(ToyLastStage())
    executor = make_executor(
        [stage],
        LinearStageMapping(pp_size=2),
        pp_rank=1,
        comm=comm,
    )
    key = TaskKey(mb=0, logical_stage=1, local_stage=0)

    executor.dispatch(RecvForwardV2(key))
    executor.dispatch(ForwardV2(key), microbatch_labels={key: torch.randn(2, 4)})
    executor.dispatch(BackwardV2(key))
    input_grad = stage.get_input_grad(key)
    losses = stage.pop_all_losses()

    assert len(comm.forward_recvs) == 1
    assert input_grad.shape == (2, 4)
    assert set(losses.keys()) == {key}
    stage.assert_clean()
    log("Remote RecvForwardV2 stages received activation")


def test_executor_dispatches_dgrad_and_wgrad_actions() -> None:
    stage = make_stage(ToyMiddleStage())
    executor = make_executor([stage], LinearStageMapping(pp_size=1))
    key = TaskKey(mb=0, logical_stage=0, local_stage=0)
    x = torch.randn(2, 4)

    stage.stage_input(key, x)
    executor.dispatch(ForwardV2(key))
    out = stage.get_output(key)
    stage.stage_grad(key, torch.randn_like(out))
    executor.dispatch(BackwardInputGradV2(key))
    input_grad = stage.get_input_grad(key)
    executor.dispatch(BackwardWeightGradV2(key))

    assert input_grad.shape == x.shape
    assert any(p.grad is not None for p in stage.model.parameters())
    stage.assert_clean()
    log("Executor dispatches BackwardInputGradV2 and BackwardWeightGradV2")


def test_executor_combined_sfrb_uses_same_peer_fast_path() -> None:
    comm = FakeComm()
    stage = make_stage(ToyFirstStage())
    executor = make_executor(
        [stage],
        LinearStageMapping(pp_size=2),
        pp_rank=0,
        comm=comm,
    )
    fwd_key = TaskKey(mb=1, logical_stage=0, local_stage=0)
    bwd_key = TaskKey(mb=0, logical_stage=0, local_stage=0)

    executor.dispatch(ForwardV2(fwd_key), microbatch_inputs={fwd_key: torch.randn(2, 4)})
    executor.dispatch(SendForwardRecvBackwardV2(fwd_key, bwd_key))

    assert [name for name, _ in comm.combined_calls] == ["sfrb"]
    assert not comm.sent_forward
    assert not comm.backward_recvs
    stage.reset()
    log("SFRB same-peer remote case uses combined comm")


def test_executor_combined_sbrf_uses_same_peer_fast_path() -> None:
    comm = FakeComm()
    stage = make_stage(ToyLastStage())
    executor = make_executor(
        [stage],
        LinearStageMapping(pp_size=2),
        pp_rank=1,
        comm=comm,
    )
    bwd_key = TaskKey(mb=0, logical_stage=1, local_stage=0)
    fwd_key = TaskKey(mb=1, logical_stage=1, local_stage=0)

    executor.dispatch(RecvForwardV2(bwd_key))
    executor.dispatch(ForwardV2(bwd_key), microbatch_labels={bwd_key: torch.randn(2, 4)})
    executor.dispatch(BackwardV2(bwd_key))
    executor.dispatch(SendBackwardRecvForwardV2(bwd_key, fwd_key))

    assert [name for name, _ in comm.combined_calls] == ["sbrf"]
    assert not comm.sent_backward
    assert len(comm.forward_recvs) == 1
    stage.reset()
    log("SBRF same-peer remote case uses combined comm")


def test_executor_rejects_deferred_actions() -> None:
    executor = make_executor(
        [make_stage(ToySingleStage())],
        LinearStageMapping(pp_size=1),
    )
    key = TaskKey(mb=0, logical_stage=0, local_stage=0)

    wait_exc = assert_raises(NotImplementedError, executor.dispatch, WaitCommV2())
    overlap_exc = assert_raises(
        NotImplementedError,
        executor.dispatch,
        OverlapForwardBackwardV2(fwd_key=key, bwd_key=key),
    )

    assert "WaitCommV2" in str(wait_exc)
    assert "OverlapForwardBackwardV2" in str(overlap_exc)
    log("Executor explicitly rejects async/overlap actions until later commits")


def main() -> None:
    tests = [
        test_executor_rejects_stage_count_mismatch,
        test_executor_rejects_non_local_task_key,
        test_executor_rejects_wrong_local_stage_in_key,
        test_executor_runs_single_stage_forward_backward,
        test_executor_local_forward_backward_handoff,
        test_executor_remote_send_forward_uses_comm,
        test_executor_remote_recv_forward_stages_input,
        test_executor_dispatches_dgrad_and_wgrad_actions,
        test_executor_combined_sfrb_uses_same_peer_fast_path,
        test_executor_combined_sbrf_uses_same_peer_fast_path,
        test_executor_rejects_deferred_actions,
    ]
    print(f"\nRunning {len(tests)} PipelineExecutor tests\n")
    for test in tests:
        print(f"[{test.__name__}]")
        test()
    print(f"\nAll {len(tests)} PipelineExecutor tests passed\n")


if __name__ == "__main__":
    main()
