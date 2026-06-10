"""Logical-to-physical pipeline stage mappings for scheduler v2.

StageMapping is the small placement contract: it describes how logical pipeline
stages are owned by physical ranks and local stage indices. BaseStageMapping
adds shared validation plus direction-aware helper queries used by v2 flows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .ir import FlowDirection, TaskKey


class StageMapping(Protocol):
    """Minimal placement interface for logical pipeline stages."""

    @property
    def num_physical_ranks(self) -> int:
        ...

    @property
    def num_logical_stages(self) -> int:
        ...

    def physical_rank(self, logical_stage: int) -> int:
        ...

    def local_stages(self, physical_rank: int) -> tuple[int, ...]:
        ...

    def local_index(self, physical_rank: int, logical_stage: int) -> int:
        ...


class BaseStageMapping:
    """Shared helpers for concrete StageMapping classes."""

    @property
    def num_physical_ranks(self) -> int:
        raise NotImplementedError

    @property
    def num_logical_stages(self) -> int:
        raise NotImplementedError

    def physical_rank(self, logical_stage: int) -> int:
        raise NotImplementedError

    def local_stages(self, physical_rank: int) -> tuple[int, ...]:
        raise NotImplementedError

    def _validate_physical_rank(self, physical_rank: int) -> None:
        if physical_rank < 0 or physical_rank >= self.num_physical_ranks:
            raise ValueError(
                f"physical_rank must be in [0, {self.num_physical_ranks}), "
                f"got {physical_rank}"
            )

    def _validate_logical_stage(self, logical_stage: int) -> None:
        if logical_stage < 0 or logical_stage >= self.num_logical_stages:
            raise ValueError(
                f"logical_stage must be in [0, {self.num_logical_stages}), "
                f"got {logical_stage}"
            )

    def local_index(self, physical_rank: int, logical_stage: int) -> int:
        self._validate_physical_rank(physical_rank)
        self._validate_logical_stage(logical_stage)

        stages = self.local_stages(physical_rank)
        try:
            return stages.index(logical_stage)
        except ValueError as exc:
            raise ValueError(
                f"logical_stage {logical_stage} is not local to "
                f"physical_rank {physical_rank}; local_stages={stages}"
            ) from exc

    def with_local_stage(self, key: TaskKey) -> TaskKey:
        self._validate_logical_stage(key.logical_stage)
        physical_rank = self.physical_rank(key.logical_stage)
        local_stage = self.local_index(physical_rank, key.logical_stage)
        if key.local_stage == local_stage:
            return key
        return TaskKey(
            mb=key.mb,
            logical_stage=key.logical_stage,
            local_stage=local_stage,
            phase=key.phase,
            direction=key.direction,
        )

    def producer_logical_stage(self, key: TaskKey) -> int | None:
        self._validate_logical_stage(key.logical_stage)

        if key.direction == FlowDirection.FORWARD:
            producer = key.logical_stage - 1
        elif key.direction == FlowDirection.REVERSE:
            producer = key.logical_stage + 1
        else:
            raise ValueError(f"unknown direction: {key.direction}")

        if producer < 0 or producer >= self.num_logical_stages:
            return None
        return producer

    def consumer_logical_stage(self, key: TaskKey) -> int | None:
        self._validate_logical_stage(key.logical_stage)

        if key.direction == FlowDirection.FORWARD:
            consumer = key.logical_stage + 1
        elif key.direction == FlowDirection.REVERSE:
            consumer = key.logical_stage - 1
        else:
            raise ValueError(f"unknown direction: {key.direction}")

        if consumer < 0 or consumer >= self.num_logical_stages:
            return None
        return consumer

    def producer_peer_rank(self, key: TaskKey) -> int | None:
        producer = self.producer_logical_stage(key)
        if producer is None:
            return None
        return self.physical_rank(producer)

    def consumer_peer_rank(self, key: TaskKey) -> int | None:
        consumer = self.consumer_logical_stage(key)
        if consumer is None:
            return None
        return self.physical_rank(consumer)

    def is_first_stage(self, key: TaskKey) -> bool:
        return self.producer_logical_stage(key) is None

    def is_last_stage(self, key: TaskKey) -> bool:
        return self.consumer_logical_stage(key) is None


@dataclass(frozen=True, slots=True)
class LinearStageMapping(BaseStageMapping):
    """Ordinary pipeline mapping: logical stage i lives on rank i."""

    pp_size: int

    def __post_init__(self) -> None:
        if self.pp_size < 1:
            raise ValueError(f"pp_size must be >= 1, got {self.pp_size}")

    @property
    def num_physical_ranks(self) -> int:
        return self.pp_size

    @property
    def num_logical_stages(self) -> int:
        return self.pp_size

    def physical_rank(self, logical_stage: int) -> int:
        self._validate_logical_stage(logical_stage)
        return logical_stage

    def local_stages(self, physical_rank: int) -> tuple[int, ...]:
        self._validate_physical_rank(physical_rank)
        return (physical_rank,)


@dataclass(frozen=True, slots=True)
class RoundRobinStageMapping(BaseStageMapping):
    """Interleaved mapping: logical stage k lives on rank k % pp_size."""

    pp_size: int
    virtual_stages: int

    def __post_init__(self) -> None:
        if self.pp_size < 1:
            raise ValueError(f"pp_size must be >= 1, got {self.pp_size}")
        if self.virtual_stages < 1:
            raise ValueError(
                f"virtual_stages must be >= 1, got {self.virtual_stages}"
            )

    @property
    def num_physical_ranks(self) -> int:
        return self.pp_size

    @property
    def num_logical_stages(self) -> int:
        return self.pp_size * self.virtual_stages

    def physical_rank(self, logical_stage: int) -> int:
        self._validate_logical_stage(logical_stage)
        return logical_stage % self.pp_size

    def local_stages(self, physical_rank: int) -> tuple[int, ...]:
        self._validate_physical_rank(physical_rank)
        return tuple(
            local_stage * self.pp_size + physical_rank
            for local_stage in range(self.virtual_stages)
        )


def validate_stage_mapping(mapping: StageMapping) -> None:
    """Validate that logical stages are covered exactly once by physical ranks."""

    owners: dict[int, int] = {}

    for physical_rank in range(mapping.num_physical_ranks):
        local = mapping.local_stages(physical_rank)
        if not local:
            raise ValueError(f"physical_rank {physical_rank} owns no logical stages")

        for local_index, logical_stage in enumerate(local):
            if logical_stage in owners:
                raise ValueError(
                    f"logical_stage {logical_stage} is owned by both "
                    f"physical_rank {owners[logical_stage]} and {physical_rank}"
                )
            owners[logical_stage] = physical_rank

            actual_physical = mapping.physical_rank(logical_stage)
            if actual_physical != physical_rank:
                raise ValueError(
                    f"inconsistent mapping for logical_stage {logical_stage}: "
                    f"local_stages says physical_rank {physical_rank}, "
                    f"physical_rank() says {actual_physical}"
                )

            actual_local_index = mapping.local_index(physical_rank, logical_stage)
            if actual_local_index != local_index:
                raise ValueError(
                    f"inconsistent local_index for logical_stage {logical_stage}: "
                    f"expected {local_index}, got {actual_local_index}"
                )

    expected = set(range(mapping.num_logical_stages))
    actual = set(owners)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"mapping does not cover logical stages exactly once: "
            f"missing={missing}, extra={extra}"
        )
