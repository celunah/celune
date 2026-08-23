# Agent mode

Agent mode is a bounded local task runner built on Persona. It is not an
arbitrary shell or computer-control agent. The runtime exposes typed,
allowlisted operations, validates every call against its schema, and routes
mutating work through approval policy.

## Task lifecycle

An agent request can move through `queued`, `idle`, `classifying`, `working`,
`planning`, `executing_tool`, `responding`, `awaiting_approval`,
`awaiting_choice`, `paused`, `interrupted`, `cancelling`, `cancelled`,
`completed`, or `failed`. Approval and choice pauses preserve the task and do
not consume an iteration. Cancellation clears pending approval/choice state.

The production limits are:

- 20 tool/planning iterations per task.
- 32,768 tokens of agent context space.
- Compaction pressure around 24,576 tokens.
- A stuck-task threshold of three repeated non-progress outcomes.

The agent can speak its final result through the standard `say()` path. It can
also be paused, resumed, cancelled, or queried by the runtime and extension
events.

## Built-in engine tools

The normal production catalog is always local and explicitly registered:

| Tool family | Calls |
| --- | --- |
| Status | `read_agent_status`, `query_status`, `query_capabilities`, `query_models`, `query_locks`, `query_audio_state`, `query_agent_task`, `run_health_check` |
| Speech | `speak`, `stop_speech`, `pause_speech` (unavailable), `resume_speech` (unavailable), `clear_speech_queue` |
| Voice and character | `set_voice`, `set_voice_prompt`, `set_playback_speed`, `set_reverb`, `set_character`, `query_character` |
| Routing and lifecycle | `set_conversation_mode`, `set_agent_mode`, `sleep`, `wake` |
| Persona memory | `remember`, `recall`, `forget`, `clear_recent_context`, `summarize_context` |
| Task control | `pause_task`, `resume_task`, `cancel_task`, `query_task`, `query_task_history` |

Important argument constraints include:

- `speak.text` is required and successful speech ends the task.
- `set_playback_speed.speed` is 0.25 through 4.0.
- `set_reverb.strength` is 0 through 1.
- `remember.importance` is 1 through 3.
- `recall.limit` is 1 through 20.
- `set_conversation_mode.mode` must be `converse`; `set_agent_mode.mode` must
  be `agent`.

Every mutating tool has `approval_required: true` in its typed schema. The
read-only tools return JSON-compatible diagnostics and do not access arbitrary
files.

## Optional local-management tools

The local filesystem and process catalog is controlled by
`agent.fs_tools: true`. It creates an unsandboxed capability boundary, so the
runtime emits a warning when it is enabled.

| Tool family | Calls | Policy |
| --- | --- | --- |
| Filesystem read | `local_current_working_directory`, `local_list_directory`, `local_file_metadata`, `local_read_text` | Absolute paths; directory listing is non-recursive; text is bounded to 1 MiB. |
| Filesystem mutation | `local_write_text`, `local_make_directory`, `local_copy` | Medium danger and approval required. |
| Filesystem destructive | `local_move`, `local_delete` | High danger; exact absolute target; recursive delete must be explicitly enabled. |
| Process diagnostics | `local_list_processes`, `local_inspect_process`, `local_system_info` | Read-only, bounded diagnostics. |
| Process control | `local_launch_process`, `local_terminate_process` | High danger; structured arguments, no shell, and identity checks. |
| Application discovery/control | `local_discover_application`, `local_running_applications`, `local_launch_application`, `local_close_application` | High-danger launch/close operations; no GUI automation. |

Local paths must be absolute and are resolved before execution. Process and
application termination requires an expected name so a stale PID cannot be
used blindly. The agent never turns a free-form prompt into a shell command.

## Needle tool selection

Production agent mode uses the Needle selector when its verified checkpoint is
available. The checkpoint is validated and prepared in an isolated cache; a
legacy JAX/Flax `needle.pkl` is not accepted as a normal production artifact.
Needle may select only registered schemas, and the runtime validates names,
argument types, approval state, and availability before execution.

## User steering and approvals

The UI displays approval and choice requests as task states. A user can answer
the request, provide a clarification, pause, resume, or cancel the task.
Extensions can subscribe to `agent_approval_requested`,
`agent_choice_requested`, `agent_task_state_changed`, and
`agent_task_finished`. See [Extensions](../interfaces/extensions.md) for the
event payloads.
