"""Unit tests for scheduler v2 StageMapping.

Run:
    PYTHONPATH=. python femtotron/test/unit/test_stage_mapping.py
"""

from __future__ import annotations

from collections.abc import Callable

from femtotron.parallel.pipeline_parallel.scheduler import (
    FlowDirection,
    LinearStageMapping,
    RoundRobinStageMapping,
    TaskKey,
    validate_stage_mapping,
)


def log(msg: str) -> None:
    print(f"  {msg}")


def assert_raises(
    expected_type: type[Exception],
    fn: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> None:
    try:
        fn(*args, **kwargs)
    except expected_type:
        return
    raise AssertionError(f"expected {expected_type.__name__} from {fn.__name__}")


def test_linear_mapping_and_direction_queries() -> None:
    mapping = LinearStageMapping(pp_size=4)

    assert mapping.num_physical_ranks == 4
    assert mapping.num_logical_stages == 4
    assert [mapping.local_stages(rank) for rank in range(4)] == [
        (0,),
        (1,),
        (2,),
        (3,),
    ]

    fwd_first = TaskKey(mb=0, logical_stage=0)
    fwd_mid = TaskKey(mb=0, logical_stage=1)
    fwd_last = TaskKey(mb=0, logical_stage=3)

    assert mapping.is_first_stage(fwd_first)
    assert not mapping.is_last_stage(fwd_first)
    assert mapping.producer_logical_stage(fwd_mid) == 0
    assert mapping.consumer_logical_stage(fwd_mid) == 2
    assert mapping.producer_peer_rank(fwd_mid) == 0
    assert mapping.consumer_peer_rank(fwd_mid) == 2
    assert mapping.is_last_stage(fwd_last)

    rev_first = TaskKey(
        mb=0,
        logical_stage=3,
        direction=FlowDirection.REVERSE,
    )
    rev_mid = TaskKey(
        mb=0,
        logical_stage=2,
        direction=FlowDirection.REVERSE,
    )
    rev_last = TaskKey(
        mb=0,
        logical_stage=0,
        direction=FlowDirection.REVERSE,
    )

    assert mapping.is_first_stage(rev_first)
    assert mapping.producer_logical_stage(rev_mid) == 3
    assert mapping.consumer_logical_stage(rev_mid) == 1
    assert mapping.producer_peer_rank(rev_mid) == 3
    assert mapping.consumer_peer_rank(rev_mid) == 1
    assert mapping.is_last_stage(rev_last)

    log("LinearStageMapping covers ordinary PP and direction-aware edges")


def test_round_robin_mapping_and_wraparound_peer() -> None:
    mapping = RoundRobinStageMapping(pp_size=4, virtual_stages=2)

    assert mapping.num_physical_ranks == 4
    assert mapping.num_logical_stages == 8
    assert [mapping.local_stages(rank) for rank in range(4)] == [
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    assert mapping.physical_rank(0) == 0
    assert mapping.physical_rank(3) == 3
    assert mapping.physical_rank(4) == 0
    assert mapping.physical_rank(7) == 3
    assert mapping.local_index(0, 0) == 0
    assert mapping.local_index(0, 4) == 1
    assert mapping.local_index(3, 7) == 1

    key3 = TaskKey(mb=0, logical_stage=3)
    key4 = TaskKey(mb=0, logical_stage=4)
    assert mapping.consumer_logical_stage(key3) == 4
    assert mapping.consumer_peer_rank(key3) == 0
    assert mapping.producer_logical_stage(key4) == 3
    assert mapping.producer_peer_rank(key4) == 3

    log("RoundRobinStageMapping covers VPP placement and wraparound peers")


def test_with_local_stage_fills_local_index() -> None:
    mapping = RoundRobinStageMapping(pp_size=4, virtual_stages=2)

    key = TaskKey(mb=0, logical_stage=5, local_stage=0)
    fixed = mapping.with_local_stage(key)

    assert fixed.mb == key.mb
    assert fixed.logical_stage == key.logical_stage
    assert fixed.local_stage == 1
    assert fixed.phase == key.phase
    assert fixed.direction == key.direction
    assert mapping.with_local_stage(fixed) is fixed

    log("with_local_stage fills the mapping-derived local stage")


def test_validate_stage_mapping_accepts_valid_mappings() -> None:
    validate_stage_mapping(LinearStageMapping(pp_size=4))
    validate_stage_mapping(RoundRobinStageMapping(pp_size=4, virtual_stages=2))

    log("validate_stage_mapping accepts built-in mappings")


def test_validate_stage_mapping_rejects_invalid_mapping() -> None:
    class DuplicateOwnerMapping:
        @property
        def num_physical_ranks(self) -> int:
            return 2

        @property
        def num_logical_stages(self) -> int:
            return 2

        def physical_rank(self, logical_stage: int) -> int:
            return 0 if logical_stage == 0 else 1

        def local_stages(self, physical_rank: int) -> tuple[int, ...]:
            return (0,) if physical_rank == 0 else (0, 1)

        def local_index(self, physical_rank: int, logical_stage: int) -> int:
            return self.local_stages(physical_rank).index(logical_stage)

    assert_raises(ValueError, validate_stage_mapping, DuplicateOwnerMapping())

    log("validate_stage_mapping rejects duplicate logical-stage owners")


def test_constructor_args_are_validated() -> None:
    for pp_size in (0, -1):
        assert_raises(ValueError, LinearStageMapping, pp_size=pp_size)

    for pp_size, virtual_stages in ((0, 1), (1, 0), (-1, 1), (1, -1)):
        assert_raises(
            ValueError,
            RoundRobinStageMapping,
            pp_size=pp_size,
            virtual_stages=virtual_stages,
        )

    log("StageMapping constructors reject invalid sizes")


def test_mapping_rejects_out_of_range_and_non_local_access() -> None:
    linear = LinearStageMapping(pp_size=4)
    assert_raises(ValueError, linear.physical_rank, -1)
    assert_raises(ValueError, linear.physical_rank, 4)
    assert_raises(ValueError, linear.local_stages, -1)
    assert_raises(ValueError, linear.local_stages, 4)

    round_robin = RoundRobinStageMapping(pp_size=4, virtual_stages=2)
    assert_raises(ValueError, round_robin.local_index, 0, 1)
    assert round_robin.physical_rank(1) != 0

    log("StageMapping rejects invalid stage and rank access")


def main() -> None:
    tests = [
        test_linear_mapping_and_direction_queries,
        test_round_robin_mapping_and_wraparound_peer,
        test_with_local_stage_fills_local_index,
        test_validate_stage_mapping_accepts_valid_mappings,
        test_validate_stage_mapping_rejects_invalid_mapping,
        test_constructor_args_are_validated,
        test_mapping_rejects_out_of_range_and_non_local_access,
    ]
    print(f"\nRunning {len(tests)} stage mapping tests\n")
    for test in tests:
        print(f"[{test.__name__}]")
        test()
    print(f"\nAll {len(tests)} stage mapping tests passed\n")


if __name__ == "__main__":
    main()
