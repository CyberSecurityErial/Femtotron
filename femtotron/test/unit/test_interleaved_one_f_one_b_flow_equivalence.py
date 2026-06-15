"""Equivalence tests for interleaved 1F1B flow actions."""

from __future__ import annotations

from pathlib import Path

import pytest

from femtotron.parallel.pipeline_parallel.action import (
    Backward,
    Forward,
    PPAction,
    RecvBackward,
    RecvForward,
    SendBackward,
    SendBackwardRecvForward,
    SendForward,
    SendForwardRecvBackward,
)
from femtotron.parallel.pipeline_parallel.interleaved_schedule import (
    decode_mb_id,
    interleaved_one_f_one_b_schedule,
)
from femtotron.parallel.pipeline_parallel.scheduler.flows.interleaved_one_f_one_b import (
    VPPAction,
    VPPBackward,
    VPPForward,
    VPPRecvBackward,
    VPPRecvForward,
    VPPSendBackward,
    VPPSendBackwardRecvForward,
    VPPSendForward,
    VPPSendForwardRecvBackward,
    VPPSingleAction,
    build_interleaved_one_f_one_b_flow,
)


def normalize_legacy_interleaved(
    actions: list[PPAction],
    *,
    num_microbatches: int,
) -> list[VPPAction]:
    normalized: list[VPPAction] = []

    def decode(eff_mb_id: int) -> tuple[int, int]:
        return decode_mb_id(eff_mb_id, num_microbatches)

    for action in actions:
        if isinstance(action, RecvForward):
            local_stage, mb = decode(action.mb_id)
            normalized.append(VPPRecvForward(local_stage, mb))
        elif isinstance(action, Forward):
            local_stage, mb = decode(action.mb_id)
            normalized.append(VPPForward(local_stage, mb))
        elif isinstance(action, SendForward):
            local_stage, mb = decode(action.mb_id)
            normalized.append(VPPSendForward(local_stage, mb))
        elif isinstance(action, RecvBackward):
            local_stage, mb = decode(action.mb_id)
            normalized.append(VPPRecvBackward(local_stage, mb))
        elif isinstance(action, Backward):
            local_stage, mb = decode(action.mb_id)
            normalized.append(VPPBackward(local_stage, mb))
        elif isinstance(action, SendBackward):
            local_stage, mb = decode(action.mb_id)
            normalized.append(VPPSendBackward(local_stage, mb))
        elif isinstance(action, SendForwardRecvBackward):
            fwd_local_stage, fwd_mb = decode(action.fwd_mb)
            bwd_local_stage, bwd_mb = decode(action.bwd_mb)
            normalized.append(
                VPPSendForwardRecvBackward(
                    fwd_local_stage=fwd_local_stage,
                    fwd_mb=fwd_mb,
                    bwd_local_stage=bwd_local_stage,
                    bwd_mb=bwd_mb,
                )
            )
        elif isinstance(action, SendBackwardRecvForward):
            bwd_local_stage, bwd_mb = decode(action.bwd_mb)
            fwd_local_stage, fwd_mb = decode(action.fwd_mb)
            normalized.append(
                VPPSendBackwardRecvForward(
                    bwd_local_stage=bwd_local_stage,
                    bwd_mb=bwd_mb,
                    fwd_local_stage=fwd_local_stage,
                    fwd_mb=fwd_mb,
                )
            )
        else:
            raise AssertionError(f"unsupported action: {action!r}")

    return normalized


def assert_real_microbatch_ids(actions: list[VPPAction], num_microbatches: int) -> None:
    for action in actions:
        if isinstance(action, VPPSingleAction):
            assert 0 <= action.mb_id < num_microbatches
        elif isinstance(action, VPPSendForwardRecvBackward):
            assert 0 <= action.fwd_mb < num_microbatches
            assert 0 <= action.bwd_mb < num_microbatches
        elif isinstance(action, VPPSendBackwardRecvForward):
            assert 0 <= action.bwd_mb < num_microbatches
            assert 0 <= action.fwd_mb < num_microbatches
        else:
            raise AssertionError(f"unsupported VPP action: {action!r}")


@pytest.mark.parametrize(
    ("pp_size", "virtual_stages", "num_microbatches"),
    [
        (1, 2, 1),
        (1, 2, 2),
        (1, 2, 4),
        (1, 3, 1),
        (1, 3, 2),
        (1, 3, 4),
        (2, 2, 2),
        (2, 2, 4),
        (2, 2, 8),
        (2, 3, 2),
        (2, 3, 4),
        (2, 3, 8),
        (4, 2, 4),
        (4, 2, 8),
        (4, 3, 4),
        (4, 3, 8),
    ],
)
def test_interleaved_one_f_one_b_flow_matches_normalized_legacy(
    pp_size: int,
    virtual_stages: int,
    num_microbatches: int,
) -> None:
    for pp_rank in range(pp_size):
        legacy = interleaved_one_f_one_b_schedule(
            num_microbatches=num_microbatches,
            pp_size=pp_size,
            pp_rank=pp_rank,
            virtual_stages=virtual_stages,
        )
        flow = build_interleaved_one_f_one_b_flow(
            num_microbatches=num_microbatches,
            pp_size=pp_size,
            pp_rank=pp_rank,
            virtual_stages=virtual_stages,
        )

        assert flow == normalize_legacy_interleaved(
            legacy,
            num_microbatches=num_microbatches,
        )
        assert_real_microbatch_ids(flow, num_microbatches)


def test_interleaved_flow_rejects_v1_with_clear_message() -> None:
    with pytest.raises(ValueError, match="build_one_f_one_b_flow"):
        build_interleaved_one_f_one_b_flow(
            num_microbatches=4,
            pp_size=2,
            pp_rank=0,
            virtual_stages=1,
        )


def test_interleaved_flow_rejects_non_divisible_microbatches() -> None:
    with pytest.raises(ValueError, match="divisible"):
        build_interleaved_one_f_one_b_flow(
            num_microbatches=5,
            pp_size=2,
            pp_rank=0,
            virtual_stages=2,
        )


def test_interleaved_flow_rejects_invalid_basic_parameters() -> None:
    with pytest.raises(ValueError, match="num_microbatches"):
        build_interleaved_one_f_one_b_flow(
            num_microbatches=0,
            pp_size=1,
            pp_rank=0,
            virtual_stages=2,
        )

    with pytest.raises(ValueError, match="pp_size"):
        build_interleaved_one_f_one_b_flow(
            num_microbatches=1,
            pp_size=0,
            pp_rank=0,
            virtual_stages=2,
        )

    with pytest.raises(ValueError, match="pp_rank"):
        build_interleaved_one_f_one_b_flow(
            num_microbatches=1,
            pp_size=2,
            pp_rank=2,
            virtual_stages=2,
        )


def test_combined_actions_can_carry_different_local_stages() -> None:
    found = False

    for pp_rank in range(4):
        actions = build_interleaved_one_f_one_b_flow(
            num_microbatches=8,
            pp_size=4,
            pp_rank=pp_rank,
            virtual_stages=3,
        )
        for action in actions:
            if isinstance(action, VPPSendForwardRecvBackward):
                found = found or action.fwd_local_stage != action.bwd_local_stage
            elif isinstance(action, VPPSendBackwardRecvForward):
                found = found or action.bwd_local_stage != action.fwd_local_stage

    assert found


def test_vpp_action_repr_is_readable() -> None:
    assert repr(VPPForward(local_stage=1, mb_id=3)) == "VF(l1,mb3)"
    assert (
        repr(
            VPPSendForwardRecvBackward(
                fwd_local_stage=0,
                fwd_mb=1,
                bwd_local_stage=2,
                bwd_mb=3,
            )
        )
        == "VSFRB(f=l0:mb1,b=l2:mb3)"
    )


def test_interleaved_flow_does_not_use_taskkey_or_effective_id_helpers() -> None:
    src = Path(
        "femtotron/parallel/pipeline_parallel/"
        "scheduler/flows/interleaved_one_f_one_b.py"
    ).read_text()

    assert "TaskKey" not in src
    assert "encode_mb_id" not in src
    assert "decode_mb_id" not in src
