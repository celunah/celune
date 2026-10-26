# SPDX-License-Identifier: Apache-2.0
"""Tests for Celune core behavior without real models or GPU work."""

from unittest import mock

import pytest

from celune.core import voice as core_voice
from celune.core import reload as core_reload
from celune.core import lifecycle as core_lifecycle

from .fixtures import CeluneCoreTestSupport


@pytest.mark.anyio
class TestCeluneAsyncRuntime(CeluneCoreTestSupport):
    """Tests for async Celune runtime entry points."""

    async def test_set_backend_async_uses_daemon_workers(self) -> None:
        """Verify async backend switching avoids asyncio's default executor."""
        celune = self._make_celune({})
        celune.loaded = True
        prepare_backend = self.patch_owner(
            core_voice, "_prepare_backend_reload", return_value=True
        )
        celune.current_voice = "nova"
        hot_reload = self.patch_owner(core_reload, "reload_backend", return_value=True)
        active_backend_name = self.patch_owner(
            core_lifecycle, "active_runtime_backend_name", return_value="mini"
        )
        run_in_daemon_thread = mock.AsyncMock(
            side_effect=lambda func, *args: func(*args)
        )

        with mock.patch(
            "celune.core.voice.run_in_daemon_thread",
            run_in_daemon_thread,
        ):
            switched = await celune.set_backend_async("mini", timeout=12.0)

        assert switched
        prepare_backend.assert_called_once_with(celune, "mini")
        hot_reload.assert_called_once_with(celune, "mini", "nova")
        active_backend_name.assert_called_once_with(celune)
        assert run_in_daemon_thread.await_count == 3

    async def test_say_async_does_not_stop_active_speech(self) -> None:
        """Verify a new async speech request does not cancel active playback."""
        celune = self._make_celune({})
        celune.force_stop_speech = mock.Mock()

        with (
            mock.patch.object(core_voice, "speech_playback_active", return_value=True),
            mock.patch(
                "celune.pipeline.submission.queue_speech_async",
                new=mock.AsyncMock(return_value=True),
            ) as queue_speech,
        ):
            assert await celune.say_async("next utterance")

        celune.force_stop_speech.assert_not_called()
        queue_speech.assert_awaited_once_with(
            celune,
            "next utterance",
            save=True,
            stream_queue=None,
            display_text=None,
        )

    def test_set_backend_stops_active_speech_before_starting_reload(self) -> None:
        """Verify backend reload requests invalidate active speech before reloading."""
        celune = self._make_celune({})
        order: list[str] = []
        celune.force_stop_speech = mock.Mock(side_effect=lambda: order.append("stop"))
        hot_reload = self.patch_owner(
            core_reload,
            "reload_backend",
            side_effect=lambda *_args: order.append("reload"),
        )

        with mock.patch(
            "celune.core.voice.threading.Thread",
            side_effect=self._immediate_thread,
        ):
            started = celune.set_backend("mini")

        assert started
        assert order == ["stop", "reload"]
        hot_reload.assert_called_once()

    async def test_wake_from_sleep_async_uses_daemon_workers(self) -> None:
        """Verify waking avoids the asyncio default executor."""
        celune = self._make_celune({})
        celune.wake_from_sleep = mock.Mock(return_value=True)
        run_in_daemon_thread = mock.AsyncMock(
            side_effect=lambda func, *args: func(*args)
        )

        with mock.patch(
            "celune.core.loading.run_in_daemon_thread",
            run_in_daemon_thread,
        ):
            woke = await celune.wake_from_sleep_async()

        assert woke
        celune.wake_from_sleep.assert_called_once_with()
        assert run_in_daemon_thread.await_count == 2
