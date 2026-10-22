# SPDX-License-Identifier: Apache-2.0
"""API authentication, session cookies, and per-client rate limits."""

from __future__ import annotations

import os
import json
import time
import base64
import secrets
import binascii
import datetime
import tempfile
import ipaddress
from pathlib import Path
from hashlib import sha256
from hmac import compare_digest
from hmac import new as hmac_new
from urllib.parse import urlsplit
from typing import Union, Optional

from fastapi import Request, WebSocket
from starlette.datastructures import Headers
from starlette.responses import Response, JSONResponse

from .constants import (
    WEBUI_AUTH_COOKIE,
    WEBUI_AUTH_CLIENT_LIMIT,
    WEBUI_AUTH_SESSION_SECONDS,
    WEBUI_GENERATION_API_NAMES,
    WEBUI_AUTH_CLIENTS_FILENAME,
)
from .session import server_session

from ..i18n import string
from ..paths import api_key_path, app_data_dir

API_KEY_PREFIX = "C-"
API_KEY_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
API_KEY_LENGTH = 25


def generate_api_key() -> str:
    """Generate a Celune API key with 25 secure base-36 characters.

    Returns:
        str: A key with the ``C-`` prefix and 25 uppercase base-36 characters.
    """
    return API_KEY_PREFIX + "".join(
        secrets.choice(API_KEY_ALPHABET) for _ in range(API_KEY_LENGTH)
    )


def _valid_generated_key(value: str) -> bool:
    """Return whether a value matches Celune's generated-key format."""
    if not value.startswith(API_KEY_PREFIX):
        return False
    suffix = value[len(API_KEY_PREFIX) :]
    return len(suffix) == API_KEY_LENGTH and all(
        character in API_KEY_ALPHABET for character in suffix
    )


def _replace_key_file(path: Path, key: str) -> None:
    """Atomically replace a malformed generated-key file."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    temporary_path = Path(temporary_name)
    try:
        if os.name != "nt":
            os.chmod(temporary_path, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(f"{key}\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def get_or_create_api_key(
    configured_key: Optional[str] = None,
    path: Optional[Path] = None,
) -> str:
    """Return a configured key or load and persist a generated key.

    Args:
        configured_key: Optional key from Celune's configuration.
        path: Optional storage path, primarily useful for isolated callers.

    Returns:
        str: The configured or persisted Celune API key.
    """
    for candidate in (configured_key, os.getenv("CELUNE_API_TOKEN")):
        if candidate is not None and candidate.strip():
            return candidate.strip()

    key_file = path or api_key_path(create_parent=True)
    key_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            key_file,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        for _attempt in range(20):
            existing = key_file.read_text(encoding="utf-8").strip()
            if _valid_generated_key(existing):
                return existing
            time.sleep(0.01)
        generated = generate_api_key()
        _replace_key_file(key_file, generated)
        return generated

    generated = generate_api_key()
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(f"{generated}\n")
        output.flush()
        os.fsync(output.fileno())
    return generated


def _clean_token(token: Optional[str]) -> Optional[str]:
    """Normalize empty token values to ``None``."""
    if token is None:
        return None
    token = token.strip()
    return token or None


def _env_auth_token() -> Optional[str]:
    """Return the API token from the environment, if configured."""
    return _clean_token(os.getenv("CELUNE_API_TOKEN"))


def configure_api_security(
    token: Optional[str] = None,
    requests_per_minute: int = 60,
) -> None:
    """Configure API authentication and rate limiting.

    Args:
        token: A required token to send requests.
        requests_per_minute: The max amount of requests per minute the user is allowed to send.
    """

    server_session.auth_token = _clean_token(token) or _env_auth_token()
    server_session.rate_limit_per_minute = max(0, requests_per_minute)
    with server_session.rate_limit_lock:
        server_session.rate_limit_hits.clear()


def resolve_api_host(token: Optional[str] = None, host: Optional[str] = None) -> str:
    """Resolve the API bind host from authentication state.

    Args:
        token: The token set up with the API.
        host: The host name or address explicitly set by the user.

    Returns:
        str: The host name or address the API is using.
    """
    if host:
        return host
    configured_token = _clean_token(token) or _env_auth_token()
    if configured_token is None:
        return "127.0.0.1"
    return "0.0.0.0"


def _request_token(request: Request) -> Optional[str]:
    """Extract the bearer or app token from a request."""
    return _headers_token(request.headers)


def _headers_token(headers: Headers) -> Optional[str]:
    """Extract the bearer or app token from request headers."""
    auth_header = headers.get("authorization", "")
    scheme, _, value = auth_header.partition(" ")
    if scheme.lower() == "bearer" and value:
        return value.strip()
    return _clean_token(headers.get("x-celune-token"))


def _headers_authenticated(headers: Headers) -> bool:
    """Return whether request headers carry the configured API token."""
    if server_session.auth_token is None:
        return True
    given = _headers_token(headers)
    return given is not None and compare_digest(given, server_session.auth_token)


def _authenticated(request: Request) -> bool:
    """Return whether the request carries the configured API token."""
    return _headers_authenticated(request.headers)


def is_browser_ui_request(request: Request) -> bool:
    """Return whether the request targets the mounted browser UI.

    Args:
        request: Incoming HTTP request to classify.

    Returns:
        bool: ``True`` when the request path points at the mounted WebUI.
    """
    path = request.url.path.rstrip("/")
    return path == "/ui" or path.startswith("/ui/")


def _is_public_api_request(request: Request) -> bool:
    """Return whether the request is safe to serve without an API token."""
    method = request.method.upper()
    path = request.url.path.rstrip("/") or "/"

    return method == "GET" and path in {"/", "/favicon.ico"}


def _client_address(connection: Union[Request, WebSocket]) -> Optional[str]:
    """Return a client IP while accepting Cloudflare identity only from its proxy."""
    client = connection.client
    if client is None:
        return None

    peer = client.host
    try:
        peer_address = ipaddress.ip_address(peer)
    except ValueError:
        return peer

    forwarded_address = connection.headers.get("cf-connecting-ip", "").strip()
    if forwarded_address:
        try:
            cloudflare_address = ipaddress.ip_address(forwarded_address)
        except ValueError:
            return peer_address.compressed
        if peer_address.is_loopback or peer_address == cloudflare_address:
            return cloudflare_address.compressed

    return peer_address.compressed


def _rate_limit_key(request: Request) -> str:
    """Return the client key used for HTTP request rate limiting."""
    address = _client_address(request) or "unknown"
    path = request.url.path.rstrip("/")
    if path == "/ui" or path.startswith("/ui/"):
        return f"webui:{address}"
    return address


def _rate_limited_client(client_key: str) -> bool:
    """Return whether a client exceeds the configured rate limit."""
    if server_session.rate_limit_per_minute <= 0:
        return False

    now = time.monotonic()
    window_start = now - 60.0

    with server_session.rate_limit_lock:
        hits = server_session.rate_limit_hits[client_key]
        while hits and hits[0] < window_start:
            hits.popleft()

        if len(hits) >= server_session.rate_limit_per_minute:
            return True

        hits.append(now)
        return False


def _rate_limited(request: Request) -> bool:
    """Return whether the request exceeds the configured rate limit."""
    return _rate_limited_client(_rate_limit_key(request))


async def _is_webui_generation_request(request: Request) -> bool:
    """Return whether a POST can start speech or voice-conversion generation."""
    if request.method.upper() != "POST":
        return False

    path = request.url.path.rstrip("/")
    if not path.startswith("/ui/"):
        return False

    if path.endswith("/queue/join"):
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        fn_index = payload.get("fn_index")
        return (
            isinstance(fn_index, int)
            and not isinstance(fn_index, bool)
            and fn_index in server_session.webui_generation_fn_indices
        )

    return any(
        path.endswith(f"/{route}/{api_name}")
        for api_name in WEBUI_GENERATION_API_NAMES
        for route in ("run", "api", "call")
    ) or any(
        path.endswith(f"/call/v2/{api_name}") for api_name in WEBUI_GENERATION_API_NAMES
    )


def _ui_session_signature(payload: bytes) -> bytes:
    """Sign a WebUI session payload with the active application key."""
    if server_session.auth_token is None:
        return b""
    return hmac_new(
        server_session.auth_token.encode("utf-8"),
        b"Celune WebUI session v1\0" + payload,
        sha256,
    ).digest()


def _webui_auth_client_cache_path() -> Path:
    """Return the app-data path for authenticated WebUI client fingerprints."""
    return app_data_dir() / WEBUI_AUTH_CLIENTS_FILENAME


def _load_webui_auth_clients_locked() -> dict[str, int]:
    """Load unexpired client fingerprints while holding their cache lock."""
    if server_session.webui_auth_clients is not None:
        return server_session.webui_auth_clients

    try:
        cached_data = json.loads(
            _webui_auth_client_cache_path().read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        cached_data = {}

    now = int(time.time())
    clients: dict[str, int] = {}
    if isinstance(cached_data, dict):
        for fingerprint, expires_at in cached_data.items():
            if (
                isinstance(fingerprint, str)
                and isinstance(expires_at, int)
                and not isinstance(expires_at, bool)
                and expires_at > now
            ):
                clients[fingerprint] = expires_at

    server_session.webui_auth_clients = clients
    return clients


def _persist_webui_auth_clients_locked(clients: dict[str, int]) -> None:
    """Write client fingerprints atomically without storing source addresses."""
    path = _webui_auth_client_cache_path()
    temporary_path = path.with_name(f"{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(
            json.dumps(clients, separators=(",", ":")),
            encoding="utf-8",
        )
        if os.name != "nt":
            os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    except OSError:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def _webui_auth_client_fingerprint(
    connection: Union[Request, WebSocket],
) -> Optional[str]:
    """Create a keyed fingerprint from a client's address and browser headers."""
    if server_session.auth_token is None:
        return None
    address = _client_address(connection)
    if address is None:
        return None

    client_headers = (
        connection.headers.get("user-agent", "")[:512],
        connection.headers.get("accept-language", "")[:256],
        connection.headers.get("accept", "")[:256],
        connection.headers.get("sec-ch-ua", "")[:256],
        connection.headers.get("sec-ch-ua-mobile", "")[:32],
        connection.headers.get("sec-ch-ua-platform", "")[:128],
    )
    client_data = "\0".join((address, *client_headers)).encode("utf-8")
    return hmac_new(
        server_session.auth_token.encode("utf-8"),
        b"Celune WebUI client cache v1\0" + client_data,
        sha256,
    ).hexdigest()


def _webui_auth_client_cached(connection: Union[Request, WebSocket]) -> bool:
    """Return whether this client has a persisted, unexpired WebUI grant."""
    fingerprint = _webui_auth_client_fingerprint(connection)
    if fingerprint is None:
        return False

    with server_session.webui_auth_clients_lock:
        clients = _load_webui_auth_clients_locked()
        expires_at = clients.get(fingerprint)
        if expires_at is None:
            return False
        if expires_at <= int(time.time()):
            del clients[fingerprint]
            _persist_webui_auth_clients_locked(clients)
            return False
        return True


def _remember_webui_auth_client(connection: Union[Request, WebSocket]) -> None:
    """Remember an authenticated client for the browser-session lifetime."""
    fingerprint = _webui_auth_client_fingerprint(connection)
    if fingerprint is None:
        return

    now = int(time.time())
    with server_session.webui_auth_clients_lock:
        clients = _load_webui_auth_clients_locked()
        clients = {
            key: expires_at for key, expires_at in clients.items() if expires_at > now
        }
        clients[fingerprint] = now + WEBUI_AUTH_SESSION_SECONDS
        if len(clients) > WEBUI_AUTH_CLIENT_LIMIT:
            clients = dict(
                sorted(
                    clients.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:WEBUI_AUTH_CLIENT_LIMIT]
            )
        server_session.webui_auth_clients = clients
        _persist_webui_auth_clients_locked(clients)


def _encode_ui_session(client_id: str, expires_at: int) -> str:
    """Create a signed, restart-safe cookie for one browser client."""
    payload = f"v1:{expires_at}:{client_id}".encode("ascii")
    encoded_payload = base64.urlsafe_b64encode(payload).rstrip(b"=")
    signature = base64.urlsafe_b64encode(_ui_session_signature(payload)).rstrip(b"=")
    return f"{encoded_payload.decode('ascii')}.{signature.decode('ascii')}"


def _ui_session_details(session: Optional[str]) -> Optional[tuple[str, int]]:
    """Return the client ID and expiry for a valid signed WebUI cookie."""
    if server_session.auth_token is None or session is None or len(session) > 512:
        return None
    try:
        encoded_payload, encoded_signature = session.split(".", 1)
        payload = base64.b64decode(
            encoded_payload.encode("ascii") + b"=" * (-len(encoded_payload) % 4),
            altchars=b"-_",
            validate=True,
        )
        signature = base64.b64decode(
            encoded_signature.encode("ascii") + b"=" * (-len(encoded_signature) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, UnicodeEncodeError, ValueError):
        return None

    if not compare_digest(signature, _ui_session_signature(payload)):
        return None

    try:
        session_version, raw_expiry, client_id = payload.decode("ascii").split(":", 2)
        expires_at = int(raw_expiry)
    except (UnicodeDecodeError, ValueError):
        return None

    now = int(time.time())
    if (
        session_version != "v1"
        or len(client_id) < 32
        or expires_at <= now
        or expires_at > now + WEBUI_AUTH_SESSION_SECONDS
    ):
        return None
    return client_id, expires_at


def _new_ui_session(client_id: Optional[str] = None) -> str:
    """Create a signed WebUI session that survives Celune restarts."""
    if server_session.auth_token is None:
        return ""
    return _encode_ui_session(
        client_id or secrets.token_urlsafe(32),
        int(time.time()) + WEBUI_AUTH_SESSION_SECONDS,
    )


def _ui_session_valid(session_id: Optional[str]) -> bool:
    """Return whether a signed WebUI session cookie is still active."""
    return _ui_session_details(session_id) is not None


def _set_ui_session_cookie(
    response: Response,
    request: Request,
    session: str,
) -> None:
    """Set a browser-scoped cookie for a validated WebUI client."""
    response.set_cookie(
        WEBUI_AUTH_COOKIE,
        session,
        max_age=WEBUI_AUTH_SESSION_SECONDS,
        path="/ui",
        secure=request.url.scheme == "https",
        httponly=True,
        samesite="strict",
    )


def _refresh_ui_session_cookie(request: Request, response: Response) -> Response:
    """Extend an active browser session near its expiry without asking again."""
    details = _ui_session_details(request.cookies.get(WEBUI_AUTH_COOKIE))
    if details is None:
        if _webui_auth_client_cached(request):
            _set_ui_session_cookie(response, request, _new_ui_session())
        return response

    client_id, expires_at = details
    if expires_at - int(time.time()) > WEBUI_AUTH_SESSION_SECONDS // 2:
        return response

    _remember_webui_auth_client(request)
    _set_ui_session_cookie(response, request, _new_ui_session(client_id))
    return response


def _webui_authenticated(connection: Union[Request, WebSocket]) -> bool:
    """Return whether a WebUI connection has an API token or active session."""
    if server_session.auth_token is None or _headers_authenticated(connection.headers):
        return True
    if _ui_session_valid(connection.cookies.get(WEBUI_AUTH_COOKIE)):
        if not _webui_auth_client_cached(connection):
            _remember_webui_auth_client(connection)
        return True
    return _webui_auth_client_cached(connection)


def _same_origin(connection: Union[Request, WebSocket]) -> bool:
    """Return whether a browser supplied origin matches this API host."""
    origin = connection.headers.get("origin")
    if origin is None:
        return False

    try:
        parsed_origin = urlsplit(origin)
        parsed_request = urlsplit(str(connection.url))
        origin_port = parsed_origin.port
        request_port = parsed_request.port
        origin_host = parsed_origin.hostname
        request_host = parsed_request.hostname
    except ValueError:
        return False

    connection_scheme = {"ws": "http", "wss": "https"}.get(
        parsed_request.scheme,
        parsed_request.scheme,
    )
    if (
        parsed_origin.username is not None
        or parsed_origin.password is not None
        or parsed_origin.path
        or parsed_origin.scheme != connection_scheme
    ):
        return False

    default_port = {"http": 80, "https": 443}.get(connection_scheme)
    return (
        origin_host is not None
        and origin_host.lower() == (request_host or "").lower()
        and (origin_port if origin_port is not None else default_port)
        == (request_port if request_port is not None else default_port)
    )


def _unauthorized_response() -> JSONResponse:
    """Build the shared unauthorized response for protected requests."""
    return JSONResponse(
        status_code=401,
        content={
            "error": "unauthorized",
            "message": string("api.unauthorized"),
        },
        headers={"WWW-Authenticate": "Bearer"},
    )


def _rate_limit_response() -> JSONResponse:
    """Build the shared rate limit response for protected requests."""
    current_time = datetime.datetime.now(datetime.UTC)
    next_minute = current_time.replace(second=0, microsecond=0) + datetime.timedelta(
        minutes=1
    )
    retry_after = (next_minute - current_time).total_seconds()
    return JSONResponse(
        status_code=429,
        content={
            "error": "ratelimit_exceeded",
            "message": string("api.rate_limit"),
        },
        headers={"Retry-After": str(retry_after)},
    )


def _websocket_authenticated(websocket: WebSocket) -> bool:
    """Return whether a WebSocket carries the configured API token."""
    if server_session.auth_token is None:
        return True

    auth_header = websocket.headers.get("authorization", "")
    scheme, _, value = auth_header.partition(" ")
    given = value.strip() if scheme.lower() == "bearer" and value else None
    if given is None:
        given = _clean_token(websocket.headers.get("x-celune-token"))
    if given is None:
        given = _clean_token(websocket.query_params.get("token"))
    return given is not None and compare_digest(given, server_session.auth_token)
