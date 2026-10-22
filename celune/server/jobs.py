# SPDX-License-Identifier: Apache-2.0
"""Speech job state and event subscriptions shared by API transports."""

from __future__ import annotations
import time
import asyncio
import threading
import contextlib
from typing import Optional
from collections import deque
from dataclasses import field, dataclass

from pydantic import BaseModel

from .session import ServerSession, server_session

from ..i18n import string
from ..constants import APP_NAME
from ..exceptions import TaskSubscriptionClosed
from ..typing.api import TaskStatus, TaskEventName

__all__ = (
    "SpeechJob",
    "TaskEvent",
    "TaskSubscription",
    "delete_expired_speech_jobs",
    "forget_speech_job",
    "publish_active_task_log",
    "publish_active_task_progress",
    "publish_task_event",
    "publish_task_progress",
    "remember_speech_job",
    "set_active_speech_task",
    "speech_job_snapshot",
    "subscribe_to_speech_job",
    "task_event_status",
    "task_status",
    "unsubscribe_from_speech_job",
    "update_speech_job",
)


class TaskEvent(BaseModel):
    """Typed event mirrored to clients watching one API task."""

    task_id: str
    event: TaskEventName
    status: TaskStatus
    message: Optional[str] = None
    severity: Optional[str] = None
    current: Optional[float] = None
    total: Optional[float] = None
    location: Optional[str] = None
    error: Optional[str] = None


@dataclass
class TaskSubscription:
    """Event-loop-owned queue with a thread-safe event publisher."""

    loop: asyncio.AbstractEventLoop
    events: asyncio.Queue[Optional[TaskEvent]] = field(
        default_factory=asyncio.Queue,
        repr=False,
    )
    closed: threading.Event = field(default_factory=threading.Event)

    def _enqueue(self, event: Optional[TaskEvent]) -> None:
        """Enqueue an event or the close sentinel on the owning loop."""
        if event is None or not self.closed.is_set():
            self.events.put_nowait(event)

    def _call_on_loop(self, event: Optional[TaskEvent]) -> None:
        """Schedule one queue mutation on the subscription's event loop."""
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is self.loop:
            self._enqueue(event)
            return

        with contextlib.suppress(RuntimeError):
            self.loop.call_soon_threadsafe(self._enqueue, event)

    def put(self, event: TaskEvent) -> None:
        """Queue one event for the subscribed WebSocket.

        Args:
            event: The typed event to deliver to the subscriber.
        """
        if not self.closed.is_set():
            self._call_on_loop(event)

    def close(self) -> None:
        """Stop waiting for events without affecting the underlying task."""
        if self.closed.is_set():
            return
        self.closed.set()
        self._call_on_loop(None)

    async def next_event(self) -> TaskEvent:
        """Wait natively on the subscription's event-loop queue.

        Returns:
            TaskEvent: The next event published for this subscription.

        Raises:
            TaskSubscriptionClosed: If the subscription was closed.
        """
        if self.closed.is_set():
            raise TaskSubscriptionClosed
        event = await self.events.get()
        if event is None:
            raise TaskSubscriptionClosed
        return event


@dataclass
class SpeechJob:
    """In-memory state for an accepted speech request."""

    status: TaskStatus
    created_at: float
    audio: Optional[bytes] = None
    error: Optional[str] = None
    events: deque[TaskEvent] = field(
        default_factory=lambda: deque(maxlen=256),
        repr=False,
    )
    subscriptions: list[TaskSubscription] = field(default_factory=list, repr=False)


def delete_expired_speech_jobs(
    now: float,
    session: ServerSession = server_session,
) -> None:
    """Remove jobs older than the in-memory job TTL."""
    expired_ids = [
        job_id
        for job_id, job in session.speech_jobs.items()
        if now - job.created_at > session.speech_job_ttl_seconds
    ]
    for job_id in expired_ids:
        session.speech_jobs.pop(job_id, None)


def remember_speech_job(
    job_id: str,
    job: SpeechJob,
    session: ServerSession = server_session,
) -> None:
    """Store one speech job and remove expired entries."""
    with session.speech_jobs_lock:
        delete_expired_speech_jobs(time.time(), session)
        session.speech_jobs[job_id] = job


def forget_speech_job(
    job_id: str,
    session: ServerSession = server_session,
) -> None:
    """Remove one speech job that was rejected before it could be observed."""
    with session.speech_jobs_lock:
        job = session.speech_jobs.pop(job_id, None)
        if job is not None:
            for subscription in job.subscriptions:
                subscription.close()
            job.subscriptions.clear()


def update_speech_job(
    job_id: str,
    *,
    status: TaskStatus,
    audio: Optional[bytes] = None,
    error: Optional[str] = None,
    session: ServerSession = server_session,
) -> None:
    """Update one speech job if it still exists."""
    with session.speech_jobs_lock:
        job = session.speech_jobs.get(job_id)
        if job is None:
            return
        job.status = status
        job.audio = audio
        job.error = error


def publish_task_event(
    job_id: str,
    event: TaskEvent,
    session: ServerSession = server_session,
) -> None:
    """Append one task event and fan it out to current subscriptions."""
    with session.speech_jobs_lock:
        job = session.speech_jobs.get(job_id)
        if job is None:
            return
        job.events.append(event)
        subscriptions = tuple(job.subscriptions)

    for subscription in subscriptions:
        subscription.put(event)


def task_status(
    job_id: str,
    session: ServerSession = server_session,
) -> Optional[TaskStatus]:
    """Return the current task status for API event association."""
    with session.speech_jobs_lock:
        job = session.speech_jobs.get(job_id)
        return None if job is None else job.status


def task_event_status(
    job_id: str,
    session: ServerSession = server_session,
) -> Optional[TaskStatus]:
    """Return a non-terminal task status suitable for callback events."""
    status = task_status(job_id, session)
    if status in {"completed", "failed", "cancelled"}:
        return None
    return status


def publish_task_progress(
    job_id: str,
    *,
    current: Optional[float] = None,
    total: Optional[float] = None,
    message: Optional[str] = None,
    session: ServerSession = server_session,
) -> None:
    """Publish a safe progress or status update for one task."""
    status = task_event_status(job_id, session)
    if status is None:
        return
    publish_task_event(
        job_id,
        TaskEvent(
            task_id=job_id,
            event="progress",
            status=status,
            current=current,
            total=total,
            message=message,
        ),
        session,
    )


def publish_active_task_log(
    message: str,
    severity: str = "info",
    session: ServerSession = server_session,
) -> None:
    """Mirror one safe Core status callback into the active speech task."""
    task_id = session.active_speech_task_id
    if task_id is None or (severity != "error" and message.startswith("[")):
        return
    status = task_event_status(task_id, session)
    if status is None:
        return
    safe_message = (
        string("status.could_not_continue", app_name=APP_NAME)
        if severity == "error"
        else message
    )
    publish_task_event(
        task_id,
        TaskEvent(
            task_id=task_id,
            event="log",
            status=status,
            message=safe_message,
            severity=severity,
        ),
        session,
    )


def set_active_speech_task(
    task_id: Optional[str],
    session: ServerSession = server_session,
) -> None:
    """Set the API task receiving Core speech callbacks."""
    session.active_speech_task_id = task_id


def publish_active_task_progress(
    current: Optional[float],
    total: Optional[float],
    session: ServerSession = server_session,
) -> None:
    """Mirror one Core progress callback into the active speech task."""
    task_id = session.active_speech_task_id
    if task_id is not None:
        publish_task_progress(
            task_id,
            current=current,
            total=total,
            session=session,
        )


def subscribe_to_speech_job(
    job_id: str,
    session: ServerSession = server_session,
) -> Optional[TaskSubscription]:
    """Subscribe to one speech job and replay its retained event history."""
    subscription = TaskSubscription(loop=asyncio.get_running_loop())
    with session.speech_jobs_lock:
        job = session.speech_jobs.get(job_id)
        if job is None:
            return None
        for event in job.events:
            subscription.put(event)
        job.subscriptions.append(subscription)
    return subscription


def unsubscribe_from_speech_job(
    job_id: str,
    subscription: TaskSubscription,
    session: ServerSession = server_session,
) -> None:
    """Detach one WebSocket subscription without changing task execution."""
    with session.speech_jobs_lock:
        job = session.speech_jobs.get(job_id)
        if job is not None:
            try:
                job.subscriptions.remove(subscription)
            except ValueError:
                pass
    subscription.close()


def speech_job_snapshot(
    job_id: str,
    session: ServerSession = server_session,
) -> Optional[SpeechJob]:
    """Return a copy of one speech job for response handling."""
    with session.speech_jobs_lock:
        delete_expired_speech_jobs(time.time(), session)
        job = session.speech_jobs.get(job_id)
        if job is None:
            return None
        return SpeechJob(
            status=job.status,
            created_at=job.created_at,
            audio=job.audio,
            error=job.error,
        )
