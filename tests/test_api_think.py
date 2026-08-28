# SPDX-License-Identifier: Apache-2.0
"""Tests for Persona think API control routes."""

import json
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import HTTPException

from celune import api
from celune.celune import Celune

from .support import CeluneTestCase


class TestApiThink(CeluneTestCase):
    """Tests for the ``/v1/think`` API endpoint."""

    def test_think_returns_accepted_when_persona_request_starts(self) -> None:
        """Verify the think endpoint accepts a request Celune can process."""
        previous_celune = api.bound_celune

        try:
            api.bound_celune = cast(
                Celune,
                SimpleNamespace(think=lambda content: content == "hello", dev=False),
            )
            response = api.think(api.ThinkRequest(content="hello"))
            payload = json.loads(bytes(response.body))

            assert response.status_code == 202
            assert payload == {"status": "accepted"}
        finally:
            api.bound_celune = previous_celune

    def test_think_returns_not_ready_when_celune_rejects_request(self) -> None:
        """Verify the think endpoint reports when Celune rejects a request."""
        previous_celune = api.bound_celune

        try:
            api.bound_celune = cast(
                Celune, SimpleNamespace(think=lambda _content: False, dev=False)
            )
            response = api.think(api.ThinkRequest(content="hello"))
            payload = json.loads(bytes(response.body))

            assert response.status_code == 409
            assert payload["error"] == "not_ready"
        finally:
            api.bound_celune = previous_celune

    def test_require_celune_still_guards_think_requests(self) -> None:
        """Verify the think endpoint fails when no Celune instance is bound."""
        previous_celune = api.bound_celune

        try:
            api.bound_celune = None
            with pytest.raises(HTTPException) as exc_info:
                api.think(api.ThinkRequest(content="hello"))

            assert exc_info.value.status_code == 503
        finally:
            api.bound_celune = previous_celune
