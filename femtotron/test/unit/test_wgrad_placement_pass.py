"""Unit tests for WGrad placement after dgrad send."""

from __future__ import annotations

import pytest

from femtotron.parallel.pipeline_parallel.action import (
    BackwardInputGrad,
    BackwardWeightGrad,
    Forward,
    SendBackward,
    SendBackwardRecvForward,
)
from femtotron.parallel.pipeline_parallel.scheduler.passes.backward_split import (
    PendingWGrad,
)
from femtotron.parallel.pipeline_parallel.scheduler.passes.wgrad_placement import (
    place_wgrad_after_dgrad_send,
)


def test_first_stage_places_wgrad_immediately_after_dgrad() -> None:
    items = [BackwardInputGrad(0), PendingWGrad(0), Forward(1)]

    assert place_wgrad_after_dgrad_send(items, is_first=True) == [
        BackwardInputGrad(0),
        BackwardWeightGrad(0),
        Forward(1),
    ]


def test_non_first_stage_places_wgrad_after_send_backward() -> None:
    items = [BackwardInputGrad(0), PendingWGrad(0), SendBackward(0)]

    assert place_wgrad_after_dgrad_send(items, is_first=False) == [
        BackwardInputGrad(0),
        SendBackward(0),
        BackwardWeightGrad(0),
    ]


def test_non_first_stage_places_wgrad_after_send_backward_recv_forward() -> None:
    items = [
        BackwardInputGrad(0),
        PendingWGrad(0),
        SendBackwardRecvForward(bwd_mb=0, fwd_mb=1),
    ]

    assert place_wgrad_after_dgrad_send(items, is_first=False) == [
        BackwardInputGrad(0),
        SendBackwardRecvForward(bwd_mb=0, fwd_mb=1),
        BackwardWeightGrad(0),
    ]


def test_wgrad_is_never_placed_before_corresponding_dgrad() -> None:
    items = [
        BackwardInputGrad(0),
        PendingWGrad(0),
        SendBackwardRecvForward(bwd_mb=0, fwd_mb=1),
    ]

    actions = place_wgrad_after_dgrad_send(items, is_first=False)

    assert actions.index(BackwardInputGrad(0)) < actions.index(BackwardWeightGrad(0))


def test_unmatched_pending_wgrad_raises() -> None:
    items = [BackwardInputGrad(0), PendingWGrad(0), Forward(1)]

    with pytest.raises(RuntimeError, match="unmatched pending W"):
        place_wgrad_after_dgrad_send(items, is_first=False)
