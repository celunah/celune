# SPDX-License-Identifier: Apache-2.0
"""Tests for Celune core behavior without real models or GPU work."""

from typing import Optional
from unittest import mock

import pytest

from celune.celune import Celune

from .support import (
    FakeGlow,
    FakeBackend,
    CeluneAsyncTestCase,
)


from .core_reload import TestCeluneCore as _TestCeluneCore


class TestCeluneCore(_TestCeluneCore):
    """Expose the split core test implementation under its original class name."""

    _cached_celune: Optional[Celune] = None
    _cached_instance_keys: frozenset[str] = frozenset()


@pytest.mark.anyio
class TestCeluneAsyncRuntime(CeluneAsyncTestCase):
    """Tests for async Celune runtime entry points."""

    @staticmethod
    def _close_celune(celune: Celune) -> None:
        """Close a test instance if it still owns the singleton slot."""
        if Celune._instance is celune:
            celune.close()

    def _make_celune(self, config: dict) -> Celune:
        """Build a Celune instance with lightweight fakes."""
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
        ):
            celune = Celune(config=config, tts_backend=FakeBackend)
            self.addCleanup(self._close_celune, celune)
            return celune

    async def test_set_backend_async_waits_for_model_ready_via_to_thread(self) -> None:
        """Verify async backend switching runs preparation and reload through to_thread."""
        celune = self._make_celune({})
        celune.loaded = True
        celune._prepare_backend_reload = mock.Mock(return_value=True)
        celune.current_voice = "nova"
        celune._hot_reload_backend = mock.Mock(return_value=True)
        celune._active_runtime_backend_name = mock.Mock(return_value="mini")
        to_thread = mock.AsyncMock(side_effect=lambda func, *args: func(*args))

        with mock.patch("celune.voice.asyncio.to_thread", to_thread):
            switched = await celune.set_backend_async("mini", timeout=12.0)

        assert switched
        celune._prepare_backend_reload.assert_called_once_with("mini")
        celune._hot_reload_backend.assert_called_once_with("mini", "nova")
        assert to_thread.await_count == 3

    def test_set_backend_stops_active_speech_before_starting_reload(self) -> None:
        """Verify backend reload requests invalidate active speech before reloading."""
        celune = self._make_celune({})
        order: list[str] = []
        celune.force_stop_speech = mock.Mock(side_effect=lambda: order.append("stop"))
        celune._hot_reload_backend = mock.Mock(
            side_effect=lambda *_args: order.append("reload")
        )

        with mock.patch(
            "celune.celune.threading.Thread",
            side_effect=TestCeluneCore._immediate_thread,
        ):
            started = celune.set_backend("mini")

        assert started
        assert order == ["stop", "reload"]

    async def test_wake_from_sleep_async_uses_to_thread(self) -> None:
        """Verify waking from sleep moves the blocking reload path off the event loop."""
        celune = self._make_celune({})
        celune.wake_from_sleep = mock.Mock(return_value=True)
        to_thread = mock.AsyncMock(side_effect=lambda func, *args: func(*args))

        with mock.patch("celune.loader.asyncio.to_thread", to_thread):
            woke = await celune.wake_from_sleep_async()

        assert woke
        celune.wake_from_sleep.assert_called_once_with()
        assert to_thread.await_count == 2
