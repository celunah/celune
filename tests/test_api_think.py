# SPDX-License-Identifier: Apache-2.0
"""Tests for Persona think API control routes."""

import json
from types import SimpleNamespace
from typing import cast
from unittest import TestCase

from fastapi import HTTPException

from celune import api
from celune.celune import Celune


class ApiThinkTests(TestCase):
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

            self.assertEqual(response.status_code, 202)
            self.assertEqual(payload, {"status": "accepted"})
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

            self.assertEqual(response.status_code, 409)
            self.assertEqual(payload["error"], "not_ready")
        finally:
            api.bound_celune = previous_celune

    def test_require_celune_still_guards_think_requests(self) -> None:
        """Verify the think endpoint fails when no Celune instance is bound."""
        previous_celune = api.bound_celune

        try:
            api.bound_celune = None
            with self.assertRaises(HTTPException) as exc_info:
                api.think(api.ThinkRequest(content="hello"))

            self.assertEqual(exc_info.exception.status_code, 503)
        finally:
            api.bound_celune = previous_celune
