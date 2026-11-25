# Agent mode

Agent mode is a bounded local task runner built on Persona. It is not an
arbitrary shell or computer-control agent. The runtime exposes typed,
allowlisted operations, validates every call against its schema, and routes
mutating work through approval policy.

Agent mode requires at least the `high` VRAM preset with a standard Persona
model. Registered smart 8B-tier Persona models require `xhigh`. At `high`, the
agent context is capped at 8,192 tokens. If a lower preset is selected, agent
routing is disabled rather than raising the configured preset automatically.

## Task lifecycle

Tasks move through planning, execution, and response. They can pause for
approval or a choice, then resume without losing progress. Cancellation clears
any pending request.

Tasks allow up to 20 planning and tool iterations by default. Set
`agent.max_loops` to change the limit. Agent context is capped at 8,192 tokens,
and `persona.compact_at` controls when older context is summarized.

Persona acknowledges each task in the active character's voice, then plans and
selects from the available tool catalog. Requests that do not match a tool
continue as ordinary Persona conversation. Follow-up requests can refer to
recent conversation, but earlier messages do not initiate work on their own.

Tool results and final responses use Celune's normal speech pipeline. The
`speak` tool waits for playback to finish before the task continues. Runtime
and extension controls own task pause, resume, cancellation, and status queries.

## Built-in engine tools

The normal production catalog is always local and explicitly registered:

| Tool family | Calls |
| --- | --- |
| Status | `read_agent_status`, `query_status`, `query_capabilities`, `query_models`, `query_locks`, `query_audio_state`, `query_agent_task`, `run_health_check` |
| Speech | `speak`, `clear_speech_queue` |
| Voice and character | `set_voice`, `set_voice_prompt`, `set_playback_speed`, `set_reverb`, `set_character`, `query_character` |
| Routing and lifecycle | `set_conversation_mode`, `set_agent_mode`, `sleep`, `wake` |
| Persona memory | `remember`, `recall`, `forget`, `clear_recent_context`, `summarize_context` |
| Task status | `query_task`, `query_task_history` |

The agent cannot interrupt active speech or pause, resume, or cancel its task.
Runtime and extension controls own those lifecycle transitions so a tool call
cannot interrupt its task's control flow.

Important argument constraints include:

- `speak.text` is required and successful speech ends the task.
- `set_playback_speed.speed` is 0.25 through 4.0.
- `set_reverb.strength` is 0 through 1.
- `remember.importance` is 1 through 3.
- `recall.limit` is 1 through 20.
- `set_conversation_mode.mode` must be `converse`. `set_agent_mode.mode` must
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
| Filesystem read | `local_current_working_directory`, `local_list_directory`, `local_file_metadata`, `local_read_text` | Absolute or working-directory-relative paths. Non-recursive listing. Text bounded to 1 MiB. |
| Filesystem mutation | `local_write_text`, `local_make_directory`, `local_copy` | Medium danger and approval required. |
| Filesystem destructive | `local_move`, `local_delete` | High danger. Exact absolute target. Recursive delete must be explicitly enabled. |
| Process diagnostics | `local_list_processes`, `local_inspect_process`, `local_system_info` | Read-only, bounded diagnostics. System information includes detected CUDA device names, capability, and memory. |
| Process control | `local_launch_process`, `local_terminate_process` | High danger. Structured arguments, no shell, and identity checks. |
| Application discovery/control | `local_discover_application`, `local_running_applications`, `local_launch_application` | High-danger launch operations. Close a verified application with `local_terminate_process`. No GUI automation. |

Read-only paths may be absolute or relative to Celune's working directory.
Changes require absolute paths so approval identifies the exact target. Process
termination requires an expected process name. The agent never runs shell
commands from free-form requests.

Process arguments are passed as a structured list without a shell. An explicit
working directory must be named in the task and exist. The agent preserves
explicit paths from your request and shows mutation targets for approval. It does
not guess ambiguous paths.

## Needle tool selection

When the verified Needle selector is available, it chooses only from registered
tool schemas. The runtime validates each call and checks availability and
approval before execution. You can name a tool directly with `Use <tool_id>`;
if multiple calls remain plausible, Celune asks you to choose. Optional
arguments may use their registered defaults, but required arguments must be
provided. If no valid call can be selected, the task ends without executing a
tool.

## User steering and approvals

Celune describes the pending action before asking for approval. Reply with a
clear answer such as `yes` or `no`; an unclear answer leaves the request pending.
Approval pauses do not count as tool iterations. Extensions can subscribe to
`agent_approval_requested`, `agent_choice_requested`,
`agent_task_state_changed`, and `agent_task_finished`. See
[Extensions](../interfaces/extensions.md) for event details.

## Verify agent availability

Run `celune test agent` to check the configured live agent path, every active
tool handler, schema validation, permissions and approvals, task lifecycle,
choices, context limits, and terminal outcomes. Its report records each check
as passed, failed, or skipped. Optional local-management checks run only when
`agent.fs_tools` is enabled. The [CLI guide](../interfaces/cli.md#commands)
describes the report and exit status.

## See also

- [CLI](../interfaces/cli.md)
- [Extensions](../interfaces/extensions.md)
