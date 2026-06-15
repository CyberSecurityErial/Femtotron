"""Equivalence tests for the scheduler 1F1B flow boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from femtotron.parallel.pipeline_parallel.action import (
    Backward,
    Forward,
    RecvBackward,
    RecvForward,
    SendBackward,
    SendBackwardRecvForward,
    SendForward,
    SendForwardRecvBackward,
)
from femtotron.parallel.pipeline_parallel.schedule import one_f_one_b_schedule
from femtotron.parallel.pipeline_parallel.scheduler.flows.one_f_one_b import (
    build_one_f_one_b_flow,
)


@pytest.mark.parametrize("pp_size", [1, 2, 4])
@pytest.mark.parametrize("num_microbatches", [1, 2, 4, 8])
def test_one_f_one_b_flow_matches_legacy(
    pp_size: int,
    num_microbatches: int,
) -> None:
    for pp_rank in range(pp_size):
        assert build_one_f_one_b_flow(
            num_microbatches=num_microbatches,
            pp_size=pp_size,
            pp_rank=pp_rank,
        ) == one_f_one_b_schedule(
            num_microbatches=num_microbatches,
            pp_size=pp_size,
            pp_rank=pp_rank,
        )


def test_one_f_one_b_flow_pp_size_one_has_no_comm_actions() -> None:
    actions = build_one_f_one_b_flow(
        num_microbatches=3,
        pp_size=1,
        pp_rank=0,
    )

    assert actions == [
        Forward(0),
        Backward(0),
        Forward(1),
        Backward(1),
        Forward(2),
        Backward(2),
    ]
    assert not any(
        isinstance(
            action,
            (
                RecvForward,
                SendForward,
                RecvBackward,
                SendBackward,
                SendForwardRecvBackward,
                SendBackwardRecvForward,
            ),
        )
        for action in actions
    )


def test_one_f_one_b_flow_preserves_legacy_small_microbatch_case() -> None:
    for pp_rank in range(4):
        assert build_one_f_one_b_flow(
            num_microbatches=1,
            pp_size=4,
            pp_rank=pp_rank,
        ) == one_f_one_b_schedule(
            num_microbatches=1,
            pp_size=4,
            pp_rank=pp_rank,
        )


def test_one_f_one_b_flow_delegates_invalid_parameter_behavior() -> None:
    with pytest.raises(ValueError, match="num_microbatches"):
        build_one_f_one_b_flow(num_microbatches=0, pp_size=1, pp_rank=0)

    with pytest.raises(ValueError, match="pp_size"):
        build_one_f_one_b_flow(num_microbatches=1, pp_size=0, pp_rank=0)

    with pytest.raises(ValueError, match="pp_rank"):
        build_one_f_one_b_flow(num_microbatches=1, pp_size=2, pp_rank=2)


def test_one_f_one_b_flow_does_not_import_taskkey() -> None:
    src = Path(
        "femtotron/parallel/pipeline_parallel/scheduler/flows/one_f_one_b.py"
    ).read_text()

    assert "TaskKey" not in src
