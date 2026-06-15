"""Equivalence tests for the ZB-lite flow boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from femtotron.parallel.pipeline_parallel.action import BackwardWeightGrad
from femtotron.parallel.pipeline_parallel.schedule import zero_bubble_schedule
from femtotron.parallel.pipeline_parallel.scheduler.flows.zero_bubble import (
    build_zero_bubble_flow,
)
from femtotron.parallel.pipeline_parallel.scheduler.passes.backward_split import (
    PendingWGrad,
)


@pytest.mark.parametrize("pp_size", [1, 2, 4])
@pytest.mark.parametrize("num_microbatches", [1, 2, 4, 8])
def test_zero_bubble_flow_matches_legacy(
    pp_size: int,
    num_microbatches: int,
) -> None:
    for pp_rank in range(pp_size):
        assert build_zero_bubble_flow(
            num_microbatches=num_microbatches,
            pp_size=pp_size,
            pp_rank=pp_rank,
        ) == zero_bubble_schedule(
            num_microbatches=num_microbatches,
            pp_size=pp_size,
            pp_rank=pp_rank,
        )


def test_zero_bubble_flow_outputs_only_executable_actions() -> None:
    actions = build_zero_bubble_flow(
        num_microbatches=4,
        pp_size=2,
        pp_rank=1,
    )

    assert actions
    assert all(not isinstance(action, PendingWGrad) for action in actions)


def test_zero_bubble_flow_contains_wgrad_actions() -> None:
    actions = build_zero_bubble_flow(
        num_microbatches=2,
        pp_size=1,
        pp_rank=0,
    )

    assert any(isinstance(action, BackwardWeightGrad) for action in actions)


def test_zb_passes_do_not_import_taskkey() -> None:
    for path in [
        "scheduler/passes/backward_split.py",
        "scheduler/passes/wgrad_placement.py",
        "scheduler/flows/zero_bubble.py",
    ]:
        src = Path(
            "femtotron/parallel/pipeline_parallel/" + path
        ).read_text()

        assert "TaskKey" not in src
