# SPDX-License-Identifier: Apache-2.0
"""Shared component ownership registry connected to Celune operations."""

from __future__ import annotations

import heapq
import asyncio
import itertools
import contextlib
import threading
import contextvars
from enum import IntEnum
from types import TracebackType
from typing import Optional, Self
from collections.abc import Callable, Generator, Sequence

from ..typing.locks import (
    ComponentLockName,
    ComponentLockOwner,
    ComponentBusyResult,
    ComponentLockAcquisition,
    ComponentLockRequirement,
)


class InferencePriority(IntEnum):
    """Priority order for inference sharing one accelerator."""

    SPEECH = 0
    PERSONA = 1


class PriorityInferenceScheduler:
    """Serialize accelerator work and let waiting speech run before Persona."""

    def __init__(self) -> None:
        """Create an idle inference scheduler."""
        self._condition = threading.Condition()
        self._waiters: list[tuple[int, int]] = []
        self._tickets = itertools.count()
        self._owner_thread: Optional[int] = None
        self._owner_depth = 0

    @contextlib.contextmanager
    def inference(self, priority: InferencePriority) -> Generator[None, None, None]:
        """Reserve the accelerator for one inference operation.

        Args:
            priority: Scheduling priority for the operation.
        """
        thread_id = threading.get_ident()
        with self._condition:
            if self._owner_thread == thread_id:
                self._owner_depth += 1
            else:
                waiter = (int(priority), next(self._tickets))
                heapq.heappush(self._waiters, waiter)
                while self._owner_thread is not None or self._waiters[0] != waiter:
                    self._condition.wait()
                heapq.heappop(self._waiters)
                self._owner_thread = thread_id
                self._owner_depth = 1

        try:
            yield
        finally:
            with self._condition:
                self._owner_depth -= 1
                if self._owner_depth == 0:
                    self._owner_thread = None
                    self._condition.notify_all()


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


async def run_in_daemon_thread[**Parameters, Result](
    function: Callable[Parameters, Result],
    *args: Parameters.args,
    **kwargs: Parameters.kwargs,
) -> Result:
    """Run a blocking operation without borrowing asyncio's default executor.

    The worker is daemonized so a canceled or shutdown-bound operation cannot
    keep an asyncio runner waiting for its default executor. Backend-owned
    cancellation should still be requested before the owning runtime is closed.

    Args:
        function: Blocking callable to execute.
        args: Positional arguments for ``function``.
        kwargs: Keyword arguments for ``function``.

    Returns:
        Result: The value returned by ``function``.
    """
    loop = asyncio.get_running_loop()
    result: asyncio.Future[Result] = loop.create_future()
    context = contextvars.copy_context()

    def complete(value: Result) -> None:
        if not result.done():
            result.set_result(value)

    def fail(error: BaseException) -> None:
        if not result.done():
            result.set_exception(error)

    def run() -> None:
        try:
            value = context.run(function, *args, **kwargs)
        except BaseException as error:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(fail, error)
        else:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(complete, value)

    threading.Thread(target=run, name="celune-async", daemon=True).start()
    return await result


__all__ = [
    "ComponentLockLease",
    "ComponentLockManager",
    "run_in_daemon_thread",
]
