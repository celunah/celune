# SPDX-License-Identifier: Apache-2.0
"""Focused tests for Celune's typed component ownership boundary."""

from typing import cast
from unittest import TestCase
from types import SimpleNamespace

from celune import pipeline
from celune.celune import Celune
from celune.locks import ComponentLockManager
from celune.agent import AgentRequest, AgentRuntime, AgentSession, AgentTaskState
from celune.typing.locks import (
    ComponentLockName,
    ComponentLockOwner,
    ComponentLockRequirement,
)

from .support import make_pipeline_engine


class ComponentLockTests(TestCase):
    """Verify atomic ownership, stale release protection, and integrations."""

    @staticmethod
    def _requirements(
        *components: ComponentLockName,
    ) -> tuple[ComponentLockRequirement, ...]:
        """Build typed requirements for one fixture operation."""
        return tuple(ComponentLockRequirement(component) for component in components)

    def test_acquisition_is_atomic_and_reports_owners(self) -> None:
        """A conflict claims nothing partially and identifies current owners."""
        manager = ComponentLockManager()
        first = ComponentLockOwner("first", session_id="session-1")
        second = ComponentLockOwner("second", session_id="session-2")
        acquired = manager.try_acquire(
            self._requirements(ComponentLockName.TTS),
            first,
        )

        self.assertTrue(acquired.acquired)
        blocked = manager.try_acquire(
            self._requirements(ComponentLockName.TTS, ComponentLockName.VLM),
            second,
        )

        self.assertFalse(blocked.acquired)
        assert blocked.busy is not None
        self.assertEqual(blocked.busy.components, (ComponentLockName.TTS,))
        self.assertEqual(blocked.busy.owners[0][1], first)
        self.assertIsNone(manager.owner_for(ComponentLockName.VLM))

    def test_lease_cleanup_and_stale_release_preserve_new_owner(self) -> None:
        """Release is idempotent and an old operation cannot clear a new claim."""
        manager = ComponentLockManager()
        first = ComponentLockOwner("first")
        second = ComponentLockOwner("second")
        acquisition, lease = manager.try_acquire_lease(
            self._requirements(ComponentLockName.AGENT),
            first,
        )
        self.assertTrue(acquisition.acquired)
        assert lease is not None
        lease.release()
        lease.release()

        reacquired = manager.try_acquire(
            self._requirements(ComponentLockName.AGENT),
            second,
        )
        self.assertTrue(reacquired.acquired)
        self.assertEqual(manager.release(first), ())
        self.assertEqual(manager.owner_for(ComponentLockName.AGENT), second)

    def test_owner_and_result_serialization_preserve_operation_metadata(self) -> None:
        """Ownership diagnostics serialize stable component and operation metadata."""
        manager = ComponentLockManager()
        owner = ComponentLockOwner(
            "operation-1",
            task_id="task-1",
            session_id="session-1",
            generation_id=3,
        )
        acquisition = manager.try_acquire(
            self._requirements(ComponentLockName.VLM),
            owner,
        )

        self.assertEqual(
            acquisition.to_json()["owner"],
            {
                "operation_id": "operation-1",
                "task_id": "task-1",
                "session_id": "session-1",
                "generation_id": 3,
            },
        )
        self.assertEqual(acquisition.to_json()["components"], ["vlm"])

    def test_release_all_clears_terminal_ownership(self) -> None:
        """Shutdown cleanup releases every component without a hidden queue."""
        manager = ComponentLockManager()
        owner = ComponentLockOwner("shutdown-operation")
        manager.try_acquire(
            self._requirements(ComponentLockName.TTS, ComponentLockName.AUDIO_PLAYBACK),
            owner,
        )

        released = manager.release_all()

        self.assertEqual(
            tuple(component for component, _ in released),
            (ComponentLockName.AUDIO_PLAYBACK, ComponentLockName.TTS),
        )
        self.assertEqual(manager.snapshot(), {})

    def test_owner_rejects_invalid_generation_metadata(self) -> None:
        """Generation identities must remain typed and non-negative."""
        with self.assertRaises(TypeError):
            ComponentLockOwner("invalid", generation_id=cast(int, "3"))
        with self.assertRaises(ValueError):
            ComponentLockOwner("invalid", generation_id=-1)

    def test_pipeline_uses_typed_owner_and_busy_result(self) -> None:
        """The legacy pipeline boolean remains backed by component ownership."""
        engine = cast(Celune, make_pipeline_engine())
        first = ComponentLockOwner("speech-operation")
        second = ComponentLockOwner("other-operation")

        acquired = pipeline.acquire_pipeline_result(engine, "speak", first)
        self.assertTrue(acquired.acquired)
        self.assertEqual(engine.component_locks.owner_for(ComponentLockName.TTS), first)

        blocked = pipeline.acquire_pipeline_result(engine, "speak", second)
        self.assertFalse(blocked.acquired)
        assert blocked.busy is not None
        self.assertIn(ComponentLockName.TTS, blocked.busy.components)
        self.assertEqual(engine._last_component_busy, blocked.busy)

        pipeline.release_pipeline(engine)
        self.assertFalse(engine.locked)
        self.assertEqual(engine.component_locks.snapshot(), {})

    def test_agent_run_returns_typed_busy_output_without_advancing_task(self) -> None:
        """An occupied agent component prevents duplicate execution safely."""
        manager = ComponentLockManager()
        other = ComponentLockOwner("other-agent", task_id="other-task")
        manager.try_acquire(self._requirements(ComponentLockName.AGENT), other)
        runtime = AgentRuntime(
            celune=cast(Celune, SimpleNamespace(component_locks=manager)),
            planner=lambda _context: {
                "tool_call": None,
                "response": "done",
                "end": True,
                "paused": False,
            },
        )
        task = runtime.create_task(
            AgentRequest(
                request="Check the status.",
                session=AgentSession(session_id="session-1"),
            ),
            task_id="blocked-agent",
        )

        output = runtime.run(task.request)

        self.assertTrue(output["paused"])
        self.assertIn("busy", output)
        self.assertEqual(task.state, AgentTaskState.IDLE)
        assert output.get("busy") is not None
        self.assertEqual(output["busy"].components, (ComponentLockName.AGENT,))
