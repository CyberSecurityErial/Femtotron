"""Unit tests for the backward split scheduler pass."""

from __future__ import annotations

import pytest

from femtotron.parallel.pipeline_parallel.action import (
    Backward,
    BackwardInputGrad,
    Forward,
    PPAction,
    SendBackward,
)
from femtotron.parallel.pipeline_parallel.scheduler.passes.backward_split import (
    PendingWGrad,
    split_backward_dw,
)


def test_split_backward_replaces_backward_with_d_and_pending_w() -> None:
    actions = [Forward(0), Backward(0), SendBackward(0)]

    assert split_backward_dw(actions) == [
        Forward(0),
        BackwardInputGrad(0),
        PendingWGrad(0),
        SendBackward(0),
    ]


def test_split_backward_preserves_non_backward_actions_and_order() -> None:
    actions = [Forward(0), SendBackward(0), Forward(1)]

    assert split_backward_dw(actions) == actions


def test_pending_wgrad_is_not_executable_action() -> None:
    pending = PendingWGrad(3)

    assert not isinstance(pending, PPAction)
    assert repr(pending) == "PW(3)"


def test_pending_wgrad_rejects_negative_microbatch_id() -> None:
    with pytest.raises(ValueError, match="mb_id"):
        PendingWGrad(-1)


def test_split_backward_does_not_need_pipeline_rank() -> None:
    actions = [Backward(2)]

    assert split_backward_dw(actions) == [
        BackwardInputGrad(2),
        PendingWGrad(2),
    ]
