# SPDX-License-Identifier: Apache-2.0
"""Process-wide state shared by the API and mounted WebUI."""

from __future__ import annotations
import threading
from collections import deque, defaultdict
from collections.abc import Callable
from dataclasses import field, dataclass
from typing import TYPE_CHECKING, Optional, Protocol

from ..i18n import string
from ..extensions.events import EventDispatcher
from ..typing.events import EventName, EventCallback

if TYPE_CHECKING:
    from .jobs import SpeechJob

    from ..celune import Celune


class ApiServerControl(Protocol):
    """Minimal server lifecycle surface required by shared runtime state."""

    should_exit: bool
    force_exit: bool


@dataclass
class ServerSession:
    """Authentication, runtime, WebUI, and speech-job state for one process."""

    bound_celune: Optional[Celune] = None
    auth_token: Optional[str] = None
    rate_limit_per_minute: int = 60
    rate_limit_lock: threading.Lock = field(default_factory=threading.Lock)
    rate_limit_hits: defaultdict[str, deque[float]] = field(
        default_factory=lambda: defaultdict(deque)
    )
    webui_auth_clients_lock: threading.Lock = field(default_factory=threading.Lock)
    webui_auth_clients: Optional[dict[str, int]] = None
    max_sfx_upload_bytes: int = 25 * 1024 * 1024
    speech_jobs_lock: threading.Lock = field(default_factory=threading.Lock)
    speech_jobs: dict[str, SpeechJob] = field(default_factory=dict)
    speech_job_ttl_seconds: int = 15 * 60
    active_speech_task_id: Optional[str] = None
    webui_log_lines: deque[tuple[str, str]] = field(
        default_factory=lambda: deque(maxlen=240)
    )
    webui_status_text: str = "Waiting for response"
    webui_status_severity: str = "info"
    webui_logs_seeded: bool = False
    webui_caption_text: str = ""
    webui_caption_progress: float = 0.0
    webui_caption_active: bool = False
    webui_progress_current: Optional[float] = None
    webui_progress_total: Optional[float] = None
    webui_resource_page: int = 0
    webui_last_resource_advance: float = 0.0
    webui_last_probed_state: Optional[str] = None
    webui_active_theme_name: str = "celune"
    webui_timed_update_sequence: int = 0
    webui_timed_update_received_at: float = 0.0
    webui_timed_update_source: str = "fallback"
    webui_timed_update_unsubscribe: Optional[Callable[[], None]] = None
    webui_event_dispatcher: Optional[EventDispatcher] = None
    webui_event_callbacks: tuple[tuple[EventName, EventCallback], ...] = ()
    webui_input_locked: bool = True
    webui_input_placeholder: str = field(
        default_factory=lambda: string("ui.wait_placeholder")
    )
    webui_voice_locked: bool = True
    webui_theme_style: str = ""
    webui_status_source: str = "probe"
    webui_status_updated_at: float = 0.0
    webui_generation_fn_indices: frozenset[int] = frozenset()
    current_api_server: Optional[ApiServerControl] = None


server_session = ServerSession()
