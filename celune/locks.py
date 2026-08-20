# SPDX-License-Identifier: MIT
"""Shared component ownership registry connected to Celune operations."""

from __future__ import annotations

import threading
from types import TracebackType
from typing import Self, Optional
from collections.abc import Sequence

from .typing.locks import (
    ComponentLockName,
    ComponentLockOwner,
    ComponentBusyResult,
    ComponentLockAcquisition,
    ComponentLockRequirement,
)


class ComponentLockLease:
    """Idempotent handle that releases one successful component acquisition."""

    def __init__(
        self,
        manager: ComponentLockManager,
        owner: ComponentLockOwner,
        components: tuple[ComponentLockName, ...],
    ) -> None:
        """Bind a lease to one atomic acquisition."""
        self._manager = manager
        self.owner = owner
        self.components = components
        self._released = False

    @property
    def released(self) -> bool:
        """Return whether this lease has already released its ownership."""
        return self._released

    def release(self) -> None:
        """Release only ownership still held by this exact operation."""
        if self._released:
            return
        self._manager.release(self.owner, self.components)
        self._released = True

    def __enter__(self) -> Self:
        """Return this lease for a context-managed operation."""
        return self

    def __exit__(
        self,
        _exc_type: Optional[type[BaseException]],
        _exc_value: Optional[BaseException],
        _traceback: Optional[TracebackType],
    ) -> None:
        """Release ownership after normal or exceptional operation exit."""
        self.release()


class ComponentLockManager:
    """Atomically reserve existing Celune resources for identified operations."""

    def __init__(self) -> None:
        """Create an empty ownership registry."""
        self._lock = threading.RLock()
        self._owners: dict[ComponentLockName, ComponentLockOwner] = {}

    def try_acquire(
        self,
        requirements: Sequence[ComponentLockRequirement],
        owner: ComponentLockOwner,
    ) -> ComponentLockAcquisition:
        """Atomically acquire all requirements or return every busy component."""
        components = self._normalize_requirements(requirements)
        with self._lock:
            unavailable = tuple(
                component
                for component in components
                if component in self._owners and self._owners[component] != owner
            )
            if unavailable:
                busy = ComponentBusyResult(
                    components=unavailable,
                    owners=tuple(
                        (component, self._owners.get(component))
                        for component in unavailable
                    ),
                )
                return ComponentLockAcquisition(owner, components, busy)

            for component in components:
                self._owners[component] = owner
            return ComponentLockAcquisition(owner, components)

    def try_acquire_lease(
        self,
        requirements: Sequence[ComponentLockRequirement],
        owner: ComponentLockOwner,
    ) -> tuple[ComponentLockAcquisition, Optional[ComponentLockLease]]:
        """Acquire requirements and return a releasable lease on success."""
        acquisition = self.try_acquire(requirements, owner)
        lease = (
            ComponentLockLease(self, owner, acquisition.components)
            if acquisition.acquired
            else None
        )
        return acquisition, lease

    def release(
        self,
        owner: ComponentLockOwner,
        components: Optional[Sequence[ComponentLockName]] = None,
    ) -> tuple[ComponentLockName, ...]:
        """Release matching ownership without allowing stale owners to clear new work."""
        requested = tuple(components) if components is not None else tuple(self._owners)
        released: list[ComponentLockName] = []
        with self._lock:
            for component in requested:
                if self._owners.get(component) == owner:
                    del self._owners[component]
                    released.append(component)
        return tuple(released)

    def owner_for(self, component: ComponentLockName) -> Optional[ComponentLockOwner]:
        """Return the current owner of one component, if it is claimed."""
        with self._lock:
            return self._owners.get(component)

    def snapshot(self) -> dict[ComponentLockName, ComponentLockOwner]:
        """Return a stable copy of current ownership for diagnostics and tests."""
        with self._lock:
            return dict(self._owners)

    def release_all(self) -> tuple[tuple[ComponentLockName, ComponentLockOwner], ...]:
        """Release every claim during shutdown or a terminal STOPPED transition."""
        with self._lock:
            released = tuple(self._owners.items())
            self._owners.clear()
            return released

    @staticmethod
    def _normalize_requirements(
        requirements: Sequence[ComponentLockRequirement],
    ) -> tuple[ComponentLockName, ...]:
        """Validate and deterministically deduplicate operation requirements."""
        components = {requirement.component for requirement in requirements}
        return tuple(sorted(components, key=lambda component: component.value))


__all__ = [
    "ComponentLockLease",
    "ComponentLockManager",
]
