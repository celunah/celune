# SPDX-License-Identifier: MIT
"""Tests for API task event streaming and cancellation."""

import asyncio
from types import SimpleNamespace
from typing import Literal, Optional, cast
from unittest import TestCase

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from celune import api
from celune.celune import Celune


class ApiTaskWebSocketTests(TestCase):
    """Verify task event history, live subscriptions, and cancellation routing."""

    def setUp(self) -> None:
        """Reset the in-memory API task registry before each test."""
        api.speech_jobs.clear()
        api.active_speech_task_id = None
        api.auth_token = None

    def tearDown(self) -> None:
        """Reset the in-memory API task registry after each test."""
        api.speech_jobs.clear()
        api.active_speech_task_id = None

    @staticmethod
    def _job(job_id: str = "task-1") -> str:
        """Create one running speech task fixture."""
        api._remember_speech_job(
            job_id,
            api.SpeechJob(status="running", created_at=0.0),
        )
        return job_id

    @staticmethod
    def _publish(
        job_id: str,
        event: api.TaskEventName,
        status: api.TaskStatus,
        *,
        message: Optional[str] = None,
        current: Optional[float] = None,
        total: Optional[float] = None,
    ) -> None:
        """Publish one typed task event fixture."""
        api._publish_task_event(
            job_id,
            api.TaskEvent(
                task_id=job_id,
                event=event,
                status=status,
                message=message,
                current=current,
                total=total,
            ),
        )

    def test_websocket_replays_events_in_order_and_closes_after_completion(
        self,
    ) -> None:
        """Verify retained task events arrive in order through one WebSocket."""
        job_id = self._job()
        self._publish(job_id, "started", "running")
        self._publish(job_id, "progress", "running", current=1.0, total=2.0)
        self._publish(job_id, "log", "running", message="Generating")
        api._update_speech_job(job_id, status="completed")
        self._publish(job_id, "completed", "completed")

        with (
            TestClient(api.api) as client,
            client.websocket_connect(f"/v1/ws/tasks/{job_id}") as websocket,
        ):
            events = [websocket.receive_json() for _ in range(4)]
            self.assertEqual(
                [event["event"] for event in events],
                ["started", "progress", "log", "completed"],
            )
            self.assertTrue(all(event["task_id"] == job_id for event in events))
            with self.assertRaises(WebSocketDisconnect):
                websocket.receive_json()

    def test_client_can_connect_after_task_started(self) -> None:
        """Verify late subscribers receive the retained start event before live events."""
        job_id = self._job()
        self._publish(job_id, "started", "running")

        with (
            TestClient(api.api) as client,
            client.websocket_connect(f"/v1/ws/tasks/{job_id}") as websocket,
        ):
            self.assertEqual(websocket.receive_json()["event"], "started")
            api._update_speech_job(job_id, status="failed", error="hidden")
            api._publish_task_event(
                job_id,
                api.TaskEvent(
                    task_id=job_id,
                    event="failed",
                    status="failed",
                    error="generation_failed",
                ),
            )
            failed = websocket.receive_json()

        self.assertEqual(failed["event"], "failed")
        self.assertNotIn("hidden", failed.values())

    def test_client_disconnect_does_not_cancel_underlying_task(self) -> None:
        """Verify closing a subscription leaves Core task state untouched."""
        job_id = self._job()
        self._publish(job_id, "started", "running")

        with (
            TestClient(api.api) as client,
            client.websocket_connect(f"/v1/ws/tasks/{job_id}") as websocket,
        ):
            websocket.receive_json()

        self.assertEqual(api.speech_jobs[job_id].status, "running")
        self.assertEqual(api.speech_jobs[job_id].subscriptions, [])

    def test_websocket_streams_failure_and_cancellation_terminal_events(self) -> None:
        """Verify failed and canceled tasks use typed terminal event names."""
        terminals: tuple[Literal["failed", "cancelled"], ...] = (
            "failed",
            "cancelled",
        )
        for index, terminal in enumerate(terminals, start=1):
            job_id = self._job(f"task-{index}")
            self._publish(job_id, "started", "running")
            api._update_speech_job(job_id, status=terminal)
            self._publish(job_id, terminal, terminal)

            with (
                TestClient(api.api) as client,
                client.websocket_connect(f"/v1/ws/tasks/{job_id}") as websocket,
            ):
                self.assertEqual(websocket.receive_json()["event"], "started")
                self.assertEqual(websocket.receive_json()["event"], terminal)

    def test_http_cancellation_calls_core_and_publishes_cancelled(self) -> None:
        """Verify explicit HTTP cancellation delegates to the Core cancellation method."""
        job_id = self._job()
        self._publish(job_id, "started", "running")
        calls: list[str] = []

        async def force_stop_speech_async() -> bool:
            calls.append("stop")
            return True

        previous_celune = api.bound_celune
        api.bound_celune = cast(
            Celune,
            SimpleNamespace(
                force_stop_speech_async=force_stop_speech_async,
            ),
        )
        try:
            response = asyncio.run(api.cancel_speech_job(job_id))
        finally:
            api.bound_celune = previous_celune

        self.assertEqual(calls, ["stop"])
        self.assertIsInstance(response, api.TaskCancelResponse)
        self.assertEqual(api.speech_jobs[job_id].status, "cancelled")
        self.assertEqual(api.speech_jobs[job_id].events[-1].event, "cancelled")
