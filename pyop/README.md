# Celune PYOP

This is Celune's detached PYOP API. It intentionally lives outside the
`celune` package so it can use a different Transformers stack.

Default runtime contract:

- host: `127.0.0.1`
- port: `2061`
- model: `lunahr/pyop-2b`
- quantization: `4bit`

Celune starts this service with the separate interpreter at `pyop/.venv`.
The user-facing `pyop` config only controls whether the companion is enabled
and whether regular UI input should use persona talkback.

Example setup:

```powershell
uv sync --project .\pyop
```

The service can also be run manually:

```powershell
.\pyop\.venv\Scripts\python pyop\run_api.py
```
