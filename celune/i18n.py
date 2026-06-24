# SPDX-License-Identifier: MIT
"""Celune internationalization stubs."""

import os
import ctypes
import contextlib
import locale as _locale  # else it gets shadowed
from typing import Optional
from types import SimpleNamespace

DEFAULT_LOCALE = "en"

STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "ui.wait_placeholder": "Please wait",
        "ui.no_voice_set": "No Voice Set",
        "webui.wait_placeholder": "Please wait",
        "webui.tutorial_placeholder": "Currently in tutorial mode",
        "webui.input_placeholder": "Enter text to speak here",
        "webui.default_voice_button": "Balanced",
        "webui.send_button": "Send",
        "ui.invalid_theme_defaulting_dark": "Invalid theme, defaulting to dark",
        "ui.sleeping_log": "{app_name} is currently sleeping. Type anything to wake up.",
        "ui.sleeping_status": "Sleeping",
        "ui.app_could_not_start": "{app_name} could not start",
        "ui.idle_status": "Idle",
        "ui.tutorial_prompt": "New to {app_name}? Type /tutorial to begin the tutorial.",
        "ui.error_signal_unavailable": "Could not play the error signal.",
        "ui.no_voices_loaded": "No voices are loaded.",
        "ui.core_engine_not_loaded": "Core engine is not loaded.",
        "headless.warn_prefix": "[WARN] ",
        "headless.error_prefix": "[ERROR] ",
        "headless.no_attached_instance": "{class_name} has no attached {app_name} instance: this will do nothing",
        "commands.no_tutorial_assets": "No tutorial assets found.",
        "commands.tutorial_playback_failed": "Tutorial playback failed: {error}",
        "commands.tutorial_failed": "Tutorial failed: {error}",
        "commands.help_header": "--- {app_name} help topics ---",
        "commands.help_available": "Available commands:",
        "commands.help_arguments": (
            "Arguments marked in <> are required, those marked in [] are optional."
        ),
        "commands.help_consumebuffer": (
            "/consumebuffer <true/false> - Make {app_name} consume text from the "
            "live buffer without pressing CTRL+ENTER."
        ),
        "commands.help_consumebuffer_caution": "Caution: This feature may interfere with typing '...'.",
        "commands.help_invoke": "/invoke <extension> [args] - Invoke a {app_name} extension by its name.",
        "commands.help_extensions": "/extensions - List currently available {app_name} extensions.",
        "commands.help_voiceprompt": (
            "/voiceprompt <prompt> - Change {app_name}'s voice prompt. This will "
            "allow you to steer her voice."
        ),
        "commands.help_voiceprompt_caution": (
            "Caution: Some prompts may cause adverse effects. Choose prompts that "
            "enhance personality, rather than replace it."
        ),
        "commands.help_speed": "/speed <speed> - Change speaking speed.",
        "commands.help_reverb": "/reverb <strength> - Change reverb strength.",
        "commands.help_backend": "/backend <name> - Hot-reload a TTS backend.",
        "commands.help_cevoice": "/cevoice <name|path> - Hot-reload a CEVOICE pack.",
        "commands.help_xvectoronly": "/xvectoronly <true/false> - Toggle Qwen3 identity-only cloning.",
        "commands.help_play": "/play <file> - Play a sound effect by path.",
        "commands.help_attach": (
            "/attach <file> [file...] - Attach one or more images or videos to the "
            "next persona reply."
        ),
        "commands.help_say": (
            "/say <text> - Speak text directly and bypass Persona for this one message."
        ),
        "commands.help_seed": (
            "/seed [seed|random] - Set or clear the seed for speech outputs, "
            "affecting pronunciation and/or prosody."
        ),
        "commands.help_tutorial": "/tutorial - Run {app_name}'s tutorial.",
        "commands.help_stop": "/stop - Terminate ongoing speech.",
        "commands.help_exit": "/exit - Exit {app_name}.",
        "commands.help_help": "/help - Display this help message.",
        "commands.usage_consumebuffer": "Usage: /consumebuffer <true/false>",
        "commands.consuming_live_input": "Now consuming from live input",
        "commands.not_consuming_live_input": "No longer consuming from live input",
        "commands.invalid_true_false_argument": "Invalid argument for '{command}', must be true/false.",
        "commands.usage_invoke": "Usage: /invoke <extension> [args]",
        "commands.extension_system_not_initialized": "Extension system not initialized.",
        "commands.extension_not_found": "Extension not found: {name}",
        "commands.extension_error": "[EXT ERROR] {error}",
        "commands.no_extensions_loaded": "No extensions loaded.",
        "commands.extensions_list": "Extensions: {names}",
        "commands.voice_prompts_unavailable": "Voice prompts are unavailable with the currently loaded model.",
        "commands.usage_voiceprompt": "Usage: /voiceprompt <prompt>",
        "commands.voice_prompt_cleared": "Voice prompt cleared.",
        "commands.voice_prompt_set": "Voice prompt set to '{prompt}'.",
        "commands.rubber_band_unavailable": "{app_name} cannot currently use Rubber Band.",
        "commands.usage_speed": "Usage: /speed <speed>",
        "commands.value_out_of_range_speed": "Value out of range. Expected 80-120%.",
        "commands.invalid_argument": "Invalid argument: {value}",
        "commands.speed_set": "Speaking speed set to {value}%.",
        "commands.usage_reverb": "Usage: /reverb <strength>",
        "commands.value_out_of_range_reverb": "Value out of range. Expected 0-100%.",
        "commands.reverb_set": "Reverb strength set to {value}%.",
        "commands.usage_backend": "Usage: /backend <name>",
        "commands.backend_already_loaded": "This backend is already loaded.",
        "commands.backend_switched": "Switched to backend: {backend_name}",
        "commands.backend_not_switched": "Backend was not switched.",
        "commands.backend_switch_failed": "Failed to switch backend: {error}",
        "commands.usage_cevoice": "Usage: /cevoice <name|path>",
        "commands.character_already_loaded": "This character is already loaded.",
        "commands.character_changed": "Character changed: {bundle}",
        "commands.character_not_switched": "Could not switch character to {bundle}.",
        "commands.character_switch_failed": "Cannot switch to this character: {error}",
        "commands.xvectoronly_qwen3_only": "This setting is only available on the Qwen3 backend.",
        "commands.usage_xvectoronly": "Usage: /xvectoronly <true/false>",
        "commands.qwen3_identity_only_cloning": "Qwen3 identity-only cloning {state}.",
        "commands.state_enabled": "enabled",
        "commands.state_disabled": "disabled",
        "commands.usage_play": "Usage: /play <path> [volume]",
        "commands.invalid_volume": "Invalid volume for '{command}', must be numeric.",
        "commands.playing_youtube_audio": "Playing YouTube audio at {volume}% volume",
        "commands.playing_audio": "Playing {path} at {volume}% volume",
        "commands.cannot_play_audio": "Cannot play this audio: {error}",
        "commands.cannot_play_file": "Cannot play this file: {error}",
        "commands.usage_attach": "Usage: /attach <file> [file...]",
        "commands.attachments_cleared": "Attachments cleared.",
        "commands.attachments_speech_only_mode": (
            "Cannot add attachments while {app_name} is running in speech-only mode."
        ),
        "commands.attachment_not_found": "Attachment not found: {path}",
        "commands.unsupported_attachment_type": "Unsupported attachment type: {path}",
        "commands.attachments_added": "Attached {names}. {count} attachments will be sent in the next pass.",
        "commands.usage_say": "Usage: /say <text>",
        "commands.say_redundant": "This command is redundant in the current operation mode.",
        "commands.say_submit_normally": "Submit inputs normally instead.",
        "commands.unmatched_ipa": "Found {count} unmatched IPA characters, output may be inaccurate.",
        "commands.custom_seed_removed": "Custom seed removed.",
        "commands.seed_range": "Seed must be between 0 and {max_value}.",
        "commands.seed_set": "Seed set to {value}.",
        "commands.tutorial_activated": "Tutorial activated. Listen to what's said to learn how to use {app_name}.",
        "commands.nothing_to_stop": "Nothing to stop.",
        "commands.unknown_command": "Unknown command: {command}. Run /help for a list of commands.",
        "cli.dependency_missing": "You do not have '{package_name}' installed.",
        "cli.dependency_required": "{app_name} requires this library to function.",
        "cli.setup_automatically": "Set up {app_name} automatically by running:",
        "cli.setup_cmd_setup_py": "    python setup.py",
        "cli.setup_with_uv": "or alternatively with uv:",
        "cli.setup_cmd_uv_sync": "    uv sync",
        "cli.install_manually": "or install the package manually:",
        "cli.setup_cmd_pip_install": "    pip install {package_name}",
        "cli.full_traceback": "for full traceback:",
        "cli.traceback_cmd_set_dev": "set CELUNE_DEV=1",
        "cli.traceback_cmd_python": "    python {script_name}",
        "cli.traceback_cmd_dev_python": "    CELUNE_DEV=1 python {script_name}",
        "cli.doctor_usage": "Usage: {program} doctor [--fix]",
        "cli.doctor_description": "Inspect {app_name}'s runtime environment without starting the app.",
        "cli.doctor_fix_help": "Use --fix to run setup.py from the repository root.",
        "cli.doctor_checking": "Checking your environment for compatibility with {app_name}...",
        "cli.doctor_hint": "hint: {hint}",
        "cli.doctor_summary": "Summary: {passes} passed, {warnings_count} warning{warning_suffix}, {failures} failed",
        "cli.doctor_ready": "Your system is ready to run {app_name}.",
        "cli.doctor_performance_impacted": "{app_name}'s performance may be impacted.",
        "cli.doctor_will_not_work": "{app_name} will not work.",
        "cli.doctor_rerun_fix": "Rerun with --fix to attempt to fix some of these problems.",
        "cli.doctor_fix_limits": (
            "Please note that this will not fix a Python version incompatibility or "
            "lack of a CUDA runtime."
        ),
        "cli.doctor_attempting_fix": "Attempting to fix fixable problems...",
        "cli.invalid_argument": "Invalid argument.",
        "cli.too_many_arguments": "Too many arguments.",
        "cli.start_usage": "Usage: {program} {command}",
        "cli.start_description": "Start {app_name}.",
        "cli.help_usage": "Usage: {program} help",
        "cli.help_description": "Display this help message.",
        "cli.help_main_usage": "Usage: {program} [command]",
        "cli.help_available_commands": "Available commands:",
        "cli.help_start": "start/run [-v]\t\t\tStart {app_name}.",
        "cli.help_config": "config [view/edit]\t\tView or edit {app_name}'s configuration.",
        "cli.help_doctor": "doctor [--fix]\t\t\tInspect the environment without starting {app_name}.",
        "cli.help_help": "help\t\t\t\tDisplay this help message.",
        "cli.help_version": "version\t\t\t\tDisplay running {app_name} version.",
        "cli.help_parameter_note": "Some commands may be used with a parameter (e.g. {program} --argument),",
        "cli.help_subcommand_note": "or with a subcommand (e.g. {program} argument).",
        "cli.help_default_start": "Providing no arguments implicitly defaults to starting {app_name}.",
        "cli.version_usage": "Usage: {program} version",
        "cli.version_description": "Display running {app_name} version.",
        "cli.modified_version_note": "Note: This is a modified version of {app_name}.",
        "cli.unknown_command_or_argument": "Unknown command or argument.",
        "cli.run_help_hint": "Run `{program} help` to list available commands.",
    },
}

STRINGS["en"].update(
    {
        "status.idle": "Idle",
        "status.speaking": "Speaking",
        "status.thinking": "Thinking",
        "status.waking_up": "Waking up",
        "status.reloading": "Reloading",
        "status.reloading_backend": "Reloading backend",
        "status.reloading_character": "Reloading character",
        "status.restoring_backend": "Restoring backend",
        "status.sleeping": "Sleeping",
        "status.initializing": "Initializing",
        "status.generating": "Generating",
        "status.normalizing": "Normalizing",
        "status.waiting_for_model": "Waiting for model",
        "status.downloading_audio": "Downloading audio",
        "status.warming_up": "Warming up",
        "status.api_starting": "Starting up",
        "status.could_not_start": "{app_name} could not start",
        "status.could_not_continue": "{app_name} could not continue",
        "status.could_not_wake": "{app_name} could not wake",
        "status.could_not_reload": "{app_name} could not reload",
        "webui.voice_ready": "Voice ready: {voice}.",
        "webui.voice_changed": "Voice changed to {voice}.",
        "webui.sleeping_log": "{app_name} is currently sleeping. Type anything to wake up.",
        "webui.must_be_running_for_commands": "{app_name} must be running to run commands.",
        "webui.command_parsing_error": "Command parsing error: {error}",
        "webui.not_available": "I'm not currently available.",
        "webui.wrong_mode": "I am not currently able to do this.",
        "webui.not_returned_from_sleep": "{app_name} has not yet returned from sleep mode.",
        "webui.busy_try_again": "I'm currently busy. Try again later.",
        "webui.error": "[WEBUI ERROR] {error}",
        "webui.upload_audio_first": "Upload or record audio before starting voice conversion.",
        "webui.conversion_only_in_vc_mode": "Audio conversion is only available in voice conversion mode.",
        "webui.cannot_convert_right_now": "I can't convert that right now.",
        "webui.cannot_change_voice_right_now": "I can't change my voice right now.",
        "webui.features_may_differ": "Usage may differ. Some {app_name} features may not be available.",
        "api.unauthorized": "Who are you? Send me an authentication token.",
        "api.rate_limit": "Please wait until you make me speak again.",
        "api.runner_started": "{app_name} API has started on http://{host}:{port}",
        "api.runner_exit_code": "API runner has exited. Exit code {code}",
        "api.could_not_start": "Could not start the API: {error}",
        "api.runner_timeout": "API runner has not responded after {seconds:.1f}s, and has timed out.",
        "api.speech_job_unknown": "I don't know that speech job.",
        "api.invalid_voice": "I don't know how to speak in that voice.",
        "api.sound_too_large": "That sound is too large for me to play.",
        "api.invalid_input": "I don't understand your input.",
        "api.cannot_play_now": "I can't play that right now.",
        "api.source_audio_too_large": "That source audio is too large for me to convert.",
        "api.could_not_convert": "I couldn't convert that.",
        "api.cannot_convert": "I can't convert that.",
        "celune.residual_temp_item": "{app_name} found a residual temporary item.",
        "celune.residual_temp_items": "{app_name} found {count} residual temporary items.",
        "celune.deleting": "Deleting...",
        "celune.persona_init_failed": "Persona could not be initialized.",
        "celune.switching_backend": "{app_name} is switching to {backend}, please wait...",
        "celune.switched_backend": "Switched backend to {backend}.",
        "celune.reload_error": "[RELOAD ERROR] {error}",
        "celune.backend_restore_failed": "Could not load this backend. The previous backend was restored.",
        "celune.reloading_character": "{app_name} is reloading the character, please stand by...",
        "celune.switched_character": "Switched to character: {character}",
        "celune.character_restore_failed": "Could not switch to this character. The previous character was restored.",
        "celune.persona_not_initialized": "Persona not initialized.",
        "celune.speech_only_mode": "Continuing in speech-only mode.",
        "celune.wake_error": "[WAKE ERROR] {error}",
        "celune.unknown_voice": "Unknown voice: {voice}",
        "celune.waiting_for_models": "Waiting for models to load...",
        "celune.voice_switch_timeout": "Timed out while switching voice.",
        "celune.reload_already_in_progress": "A backend or character reload is already in progress.",
        "celune.unknown_backend": "Unknown backend: {backend} (available: {available})",
        "celune.backend_switch_timeout": "Timed out while switching backends.",
        "celune.voice_pack_not_found": "Voice pack not found: {bundle}",
        "celune.character_switch_timeout": "Timed out while switching characters.",
        "celune.ready_wait_timeout": "Timed out while waiting to become ready.",
        "celune.ready_wait_reason": "A possible reason for this may be a model download or high GPU activity.",
        "celune.ready_wait_not_fatal": "This is not a fatal error, the utterance may be retried.",
        "celune.model_unloaded_while_waiting": "Model was unloaded while waiting to become ready.",
        "celune.playback_idle_timeout": "Timed out while waiting for playback pipeline to become idle.",
        "celune.reloading": "{app_name} is reloading, please stand by...",
        "celune.rewarming_up": "Rewarming up...",
        "celune.voice_loaded": "Voice {voice} loaded.",
        "celune.no_voices_loaded": "No voices were loaded.",
        "celune.current_character_default": "Current character: {character} (default)",
        "celune.current_character": "Current character: {character}",
        "celune.current_vram_preset": "Current VRAM preset: {preset}",
        "celune.no_vc_backend": "No voice conversion backend is available.",
        "celune.no_voices_loaded_short": "No voices loaded",
        "celune.no_valid_vc_backend": "No valid voice conversion backend",
        "celune.ready_for_vc": "Ready to accept voice conversions.",
        "celune.all_voices_available": "All voices are available.",
        "celune.default_model_load_failed": "{app_name} could not load the default model.",
        "celune.default_model_failed_short": "Default model failed to load",
        "celune.initializing_persona": "Initializing Persona...",
        "celune.persona_initialized": "Persona initialized.",
        "celune.warmup_failed": "[WARMUP] Warmup failed.",
        "celune.personas_unavailable": "Personas are unavailable. {app_name} is operating in speech-only mode.",
        "celune.no_api_token": "No API token set. {app_name} API will bind only to the local network.",
        "celune.api_port_invalid": "{app_name} API port ({port}) is invalid, will use 2060 instead.",
        "celune.api_port_out_of_range": "{app_name} API port ({port}) is out of range, will use 2060 instead.",
        "celune.api_rate_limit_invalid": "{app_name} API rate limit ({rate}) is invalid, using 60/min.",
        "celune.port_unavailable": "Port {port} is unavailable.",
        "celune.api_unavailable": "{app_name} API will not be available.",
        "celune.required_package_missing": "A required package ({package}) isn't installed.",
        "celune.internal_error": "An internal error occurred: {error}",
        "celune.package_import_failed": "Package import failed: {error}",
        "celune.normalizer_loaded": "Normalizer loaded.",
        "celune.normalizer_error": "[NORMALIZER ERROR] {error}",
        "celune.normalizer_failed": "Normalizer failed to load.",
        "celune.normalization_unavailable": "Normalization will not be available.",
        "celune.loading_normalizer": "Loading normalizer {model_id} on {device}...",
        "celune.warmup_start": "[WARMUP] Warming up...",
        "celune.warmup_error": "[WARMUP ERROR] {error}",
        "celune.warmup_failed_app": "{app_name} could not warm up",
        "celune.tokens_to_normalize": "Tokens to normalize: {count}",
        "celune.input_too_long_to_normalize": "Input is too long to normalize.",
        "celune.normalizer_returned_no_tokens": "Normalizer returned no tokens.",
        "celune.normalizer_bad_output": "Normalizer did not produce normal output.",
        "celune.normalized_text": "Normalized text: {text}",
        "celune.normalization_took": "Normalization took {seconds} seconds.",
        "celune.normalization_error": "[NORMALIZATION ERROR] {error}",
        "celune.text_input_unavailable_vc": "Text input is unavailable in voice conversion mode.",
        "celune.speech_input_disabled_tutorial": "Speech input is disabled during the tutorial.",
        "celune.cannot_think_sleeping": "Cannot think while {app_name} is sleeping.",
        "celune.busy_thinking": "Tried to think while {app_name} was busy.",
        "celune.app_sleeping": "{app_name} is currently sleeping",
        "celune.app_busy": "{app_name} is currently busy",
        "celune.say_instead": "Will say the input instead.",
        "celune.audio_conversion_unavailable": "Audio conversion is unavailable outside voice conversion mode.",
        "celune.not_possible": "Not possible",
        "pipeline.ttfp_seconds": "TTFP: {seconds} seconds",
        "pipeline.forcefully_stopping_speech": "Forcefully stopping speech.",
        "pipeline.busy_action": "Tried to {action} while {app_name} was busy.",
        "pipeline.yt_dlp_missing": "yt-dlp is not installed, cannot play YouTube audio.",
        "pipeline.yt_dlp_required": "yt-dlp is required for YouTube playback",
        "pipeline.youtube_download_start": "[SFX] Downloading audio from {url}...",
        "pipeline.download_failed": "Could not download audio.",
        "pipeline.download_youtube_failed_short": "Could not download YouTube audio",
        "pipeline.download_timeout": "Timed out downloading audio.",
        "pipeline.downloader_no_file": "Downloader returned no file.",
        "pipeline.persona_not_connected": "Persona system is not connected.",
        "pipeline.persona_request_failed": "Persona system request failed: {error}",
        "pipeline.persona_empty_response": "Persona system returned an empty response.",
        "pipeline.vc_backend_unconfigured": "Voice conversion backend is not configured.",
        "pipeline.vc_reference_load_failed": "Could not load reference audio for voice conversion: {error}",
        "pipeline.cannot_speak_sleeping": "Cannot speak while {app_name} is sleeping.",
        "pipeline.speak_waiting_reload": "Speak request is waiting for model reload to finish.",
        "pipeline.received_unsupported_language": "Received unsupported input in the following language: {language}",
        "pipeline.may_not_say_properly": "{app_name} may not say the input properly.",
        "pipeline.april_fools": "We are about to do a funny!",
        "pipeline.sample_rate_length": "Sample rate: {sample_rate} Hz, length: {seconds} seconds",
        "pipeline.playing_label": "Playing {label}",
        "pipeline.cannot_find_sound": "{app_name} cannot find {path}.",
        "pipeline.sfx_format_unsupported": "{app_name} does not support SFX in this format.",
        "pipeline.supported_formats": "Supported formats: {formats}",
        "pipeline.exiting": "Exiting...",
        "pipeline.nothing_to_say": "Nothing to say",
        "pipeline.rubber_band_unavailable": "Rubber Band is unavailable, speed controls disabled.",
        "pipeline.token_limit_reached": (
            "Generation reached the token limit before completion. Output may be "
            "truncated or sound incorrect."
        ),
        "pipeline.generation_summary": "[GEN] {speech_seconds} seconds, took {generation_seconds} seconds",
        "pipeline.generation_speed": "Speed: x{speed}",
        "pipeline.ttfc_ms": "TTFC: {milliseconds} ms",
        "pipeline.generation_done": "[GEN] done",
        "pipeline.silent_regenerating": "Previous utterance was silent, regenerating...",
        "pipeline.may_be_silent": "This utterance may be unexpectedly silent.",
        "pipeline.outputs_path_creating": "Outputs path not found, creating...",
        "pipeline.outputs_create_failed": "Cannot create outputs directory, not saving FLAC output: {error}",
        "pipeline.flac_save_failed": "Could not save FLAC output: {error}",
        "pipeline.gen_error": "[GEN ERROR] {error}",
        "pipeline.could_not_generate": "{app_name} could not generate the input",
        "pipeline.not_ready_app": "{app_name} is not currently ready",
        "pipeline.audio_stream_init_failed": "{app_name} could not initialize the audio stream.",
        "pipeline.no_audio_device": "No suitable audio device is available.",
        "pipeline.no_audio_devices_short": "No suitable audio devices",
        "pipeline.just_type": "Just type. {choice}",
        "pipeline.ready_to_speak": "Ready to speak.",
        "pipeline.vram_low": (
            "{app_name} is running out of VRAM. Check the bottom right of "
            "{app_name}'s window to learn more."
        ),
        "pipeline.close_memory_apps": "Please close any memory-resident applications to improve performance.",
        "pipeline.playback_error": "Playback error",
        "extensions.autostart_once": "[Core] Cannot autostart Celune extensions more than one time.",
        "extensions.autostart_all_deprecated": (
            "CeluneExtensionManager.autostart_all() is deprecated, please use "
            "@celune.subscribe('ready') in your extensions instead"
        ),
        "extensions.autostart_failed": "[Core] Could not autostart {name}: {error}",
        "extensions.invoke_failed": "[Core] Failed to invoke '{name}': {error}",
        "extensions.folder_not_found": "[Core] Extension folder not found: {path}",
        "extensions.unavailable": "Extensions will not be available.",
        "extensions.path_not_directory": "[Core] Extension path is not a directory: {path}",
        "extensions.spec_load_failed": "[Core] Could not load spec for: {name}",
        "extensions.import_failed": "[Core] Failed to import '{name}': {error}",
        "extensions.register_failed": "[Core] Failed to register '{name}' from '{file_name}': {error}",
        "extensions.not_extension_skipping": "[Core] {file_name} is not a Celune extension, skipping",
        "extensions.autostart_deprecated": (
            "CeluneExtension.autostart() is deprecated, please use "
            "@celune.subscribe('ready') instead"
        ),
        "cli.fix_failed": "Failed to fix fixable problems: {error}",
        "cli.config_not_created": "{app_name} configuration has not been created yet.",
        "cli.run_once_to_create_config": "Run {app_name} at least once to create it.",
        "cli.current_config": "Current {app_name} configuration:",
        "cli.config_could_not_be_read": "{app_name} configuration could not be read.",
        "cli.config_usage": "Usage: {program} config [view/edit]",
        "cli.config_description": "View or edit {app_name}'s configuration.",
        "cli.no_argument_given": "No argument given.",
        "cli.running_from_ide": "{app_name} is running from {ide}.",
        "cli.ide_terminals_differ": "Some IDE terminals may behave differently from a normal terminal.",
        "cli.config_created": "{app_name} configuration has been created.",
        "cli.config_updated_defaults": "{app_name} configuration has been updated with new defaults.",
        "cli.launcher_apply_artifact": "{app_name} will close so the launcher can apply the latest artifact.",
        "cli.update_info_incomplete": "update information is incomplete",
        "cli.update_found": "New update found.",
        "cli.update_prompt": "Do you want to update?",
        "cli.update_version_summary": (
            "You are running {app_name} {local_version} ({local_revision}), latest "
            "version is {latest_label} ({latest_revision})."
        ),
        "cli.update_choice_yes": "Yes, update now",
        "cli.update_choice_no": "No, continue as is",
        "cli.updating": "Updating {app_name}...",
        "cli.update_failed": "{app_name} could not update: {detail}",
        "cli.continuing_current_version": "Continuing with the current version.",
        "cli.update_success_restart": "{app_name} updated successfully. Restart {app_name} to apply changes.",
        "cli.no_ansi": "This terminal does not support ANSI.",
        "cli.request_refresh_binaries": "Requesting the launcher to refresh the packaged binaries...",
        "cli.apply_update_noninteractive": "Attempting to apply update non-interactively...",
        "cli.not_via_launcher": "{app_name} is not being launched via the {app_name} launcher.",
        "cli.suppress_message_run_with": "To suppress this message, run {app_name} with:",
        "cli.or_set_env_var": "or set the following environment variable:",
        "cli.already_running": "{app_name} is already running.",
        "cli.could_not_initialize": "{app_name} could not initialize.",
        "cli.running_headless": "{app_name} is running in headless mode.",
        "cli.headless_extensions_only": "While in this mode, input is only possible via {app_name} extensions.",
        "cli.headless_press_ctrl_c": "Press CTRL+C in this window to stop it.",
        "cli.cannot_start_normal_mode": "{app_name} cannot start in normal mode.",
        "cli.hint": "Hint:",
        "cli.try_another_terminal": "Try using another terminal application.",
        "cli.internal_error_running": "An internal error occurred while {app_name} was running.",
        "cli.no_error_description": "no error description",
        "cli.full_traceback_title": "For full traceback:",
        "cli.additional_debugging": "additional debugging:",
        "cli.set_dev_true": "Set 'dev: true' in config.yaml",
        "cli.celine_day_1": "I sense the presence of... her.",
        "cli.celine_day_2": "I would rather not.",
        "cli.try_again_tomorrow": "Try again tomorrow.",
        "cli.apply_update_usage": "Usage: celune __apply_update <parent-pid> <launcher-path> [args...]",
        "cli.invalid_launcher_pid": "Invalid launcher PID.",
        "cli.apply_launcher_update_failed": "{app_name} could not apply the launcher update: {error}",
    }
)


def _normalize_locale_name(locale_name: Optional[str]) -> str:
    """Normalize a locale label into a lookup-friendly language code."""
    if not locale_name:
        return DEFAULT_LOCALE

    return locale_name.replace("_", "-").split(".", maxsplit=1)[0].lower()


def _locale_candidates(locale_name: Optional[str]) -> list[str]:
    """Return locale lookup candidates from most to least specific."""
    normalized = _normalize_locale_name(locale_name)
    candidates = [normalized]

    if "-" in normalized:
        candidates.append(normalized.split("-", maxsplit=1)[0])

    if DEFAULT_LOCALE not in candidates:
        candidates.append(DEFAULT_LOCALE)

    return candidates


def get_system_locale() -> str:
    """Get the current system locale, falling back to English if unavailable.

    Returns:
        str: The detected locale code, or ``"en"`` when no locale can be found.
    """
    lang, _ = _locale.getlocale()
    if lang:
        return lang

    if os.name == "nt":
        with contextlib.suppress(Exception):
            windll = getattr(ctypes, "windll", SimpleNamespace()).kernel32
            lang_code = windll.GetUserDefaultUILanguage()
            return _locale.windows_locale.get(lang_code, "en")

    lang = os.environ.get("LANG")
    if lang:
        return lang.split(".")[0]

    return "en"


_current_locale = get_system_locale()


def set_locale(locale: str) -> None:
    """Set Celune's active locale.

    Args:
        locale: The locale code to store as the current language selection.
    """
    global _current_locale
    _current_locale = locale


def get_locale() -> str:
    """Get Celune's current locale setting.

    Returns:
        str: The currently configured locale code.
    """
    return _current_locale


def string(key: str, locale: Optional[str] = None, **kwargs) -> str:
    """Get an internationalized string for the selected language.

    Args:
        key: The translation key to look up.
        locale: An optional locale override. When omitted, the current locale is used.
        kwargs: Optional format values interpolated into the resolved string.

    Returns:
        str: The translated string, or the key itself when no translation exists.
    """
    text = None
    for lang in _locale_candidates(locale or _current_locale):
        text = STRINGS.get(lang, {}).get(key)
        if text is not None:
            break

    if text is None:
        text = key

    if kwargs:
        return text.format(**kwargs)

    return text
