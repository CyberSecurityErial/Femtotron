"""Unit tests for scheduler v2 IR.

Run:
    PYTHONPATH=. python femtotron/test/unit/test_scheduler_ir.py
"""

from __future__ import annotations

from femtotron.parallel.pipeline_parallel.action import (
    Backward,
    Forward,
    SendForwardRecvBackward,
)
from femtotron.parallel.pipeline_parallel.scheduler.ir import (
    BackwardInputGradV2,
    BackwardV2,
    BackwardWeightGradV2,
    FlowDirection,
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


def log(msg: str) -> None:
    print(f"  {msg}")


def test_task_key_is_hashable_dict_key() -> None:
    key = TaskKey(mb=3, logical_stage=5, local_stage=1, phase=2)
    reverse_key = TaskKey(
        mb=3,
        logical_stage=5,
        local_stage=1,
        phase=2,
        direction=FlowDirection.REVERSE,
    )
    values = {key: "forward", reverse_key: "reverse"}

    assert values[key] == "forward"
    assert values[reverse_key] == "reverse"
    assert key != reverse_key
    log("TaskKey is hashable and direction participates in identity")


def test_task_key_short_is_stable() -> None:
    key = TaskKey(
        mb=7,
        logical_stage=9,
        local_stage=2,
        phase=1,
        direction=FlowDirection.REVERSE,
    )

    assert key.short() == "mb=7,ls=9,local=2,phase=1,dir=reverse"
    assert repr(key) == "TaskKey(mb=7,ls=9,local=2,phase=1,dir=reverse)"
    log("TaskKey.short() and repr are stable")


def test_v2_action_repr_is_readable() -> None:
    fwd = TaskKey(mb=0, logical_stage=1)
    bwd = TaskKey(mb=0, logical_stage=1, phase=1, direction=FlowDirection.REVERSE)

    actions = [
        ForwardV2(fwd),
        BackwardV2(bwd),
        BackwardInputGradV2(bwd),
        BackwardWeightGradV2(bwd),
        RecvForwardV2(fwd),
        SendForwardV2(fwd),
        RecvBackwardV2(bwd),
        SendBackwardV2(bwd),
        SendForwardRecvBackwardV2(fwd_key=fwd, bwd_key=bwd),
        SendBackwardRecvForwardV2(bwd_key=bwd, fwd_key=fwd),
        OverlapForwardBackwardV2(fwd_key=fwd, bwd_key=bwd),
        WaitCommV2(),
    ]
    rendered = [repr(action) for action in actions]

    assert rendered == [
        "F2(mb=0,ls=1,local=0,phase=0,dir=forward)",
        "B2(mb=0,ls=1,local=0,phase=1,dir=reverse)",
        "D2(mb=0,ls=1,local=0,phase=1,dir=reverse)",
        "W2(mb=0,ls=1,local=0,phase=1,dir=reverse)",
        "RF2(mb=0,ls=1,local=0,phase=0,dir=forward)",
        "SF2(mb=0,ls=1,local=0,phase=0,dir=forward)",
        "RB2(mb=0,ls=1,local=0,phase=1,dir=reverse)",
        "SB2(mb=0,ls=1,local=0,phase=1,dir=reverse)",
        (
            "SFRB2(fwd=mb=0,ls=1,local=0,phase=0,dir=forward,"
            "bwd=mb=0,ls=1,local=0,phase=1,dir=reverse)"
        ),
        (
            "SBRF2(bwd=mb=0,ls=1,local=0,phase=1,dir=reverse,"
            "fwd=mb=0,ls=1,local=0,phase=0,dir=forward)"
        ),
        (
            "OFB2(fwd=mb=0,ls=1,local=0,phase=0,dir=forward,"
            "bwd=mb=0,ls=1,local=0,phase=1,dir=reverse)"
        ),
        "WaitComm2()",
    ]
    log("V2 action reprs are compact and readable")


def test_legacy_action_repr_is_unchanged() -> None:
    assert repr(Forward(2)) == "F(2)"
    assert repr(Backward(2)) == "B(2)"
    assert repr(SendForwardRecvBackward(fwd_mb=3, bwd_mb=1)) == "SFRB(3,1)"
    log("legacy PPAction reprs are unchanged")


def main() -> None:
    tests = [
        test_task_key_is_hashable_dict_key,
        test_task_key_short_is_stable,
        test_v2_action_repr_is_readable,
        test_legacy_action_repr_is_unchanged,
    ]
    print(f"\nRunning {len(tests)} scheduler IR tests\n")
    for test in tests:
        print(f"[{test.__name__}]")
        test()
    print(f"\nAll {len(tests)} scheduler IR tests passed\n")


if __name__ == "__main__":
    main()
