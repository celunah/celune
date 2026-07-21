-- SPDX-License-Identifier: MIT

-- What should this extension be called?
EXTENSION_NAME = "Test"

subscribe("ready", function(event)
	-- What Lua version is Celune using?
    celune.log("Celune Lua runtime is running this extension on ".._VERSION)
    celune.sleep(5)

    -- Write to Celune's logs.
    celune.log("Log test")
    celune.sleep(1)

    -- Set Celune's status.
    celune.status("Status test")
    celune.sleep(5)

	-- You can also set other severities.
    celune.status("Status test (warning)", "warning")
    celune.sleep(5)
    celune.status("Status test (error)", "error")
    celune.sleep(5)
    celune.status("Status test (unknown)", "invalid")
    celune.sleep(5)

    -- Make Celune say something.
    celune.say("Speaking with default voice.")
    celune.sleep(1)

    -- Set Celune's voice and speak again.
    celune.set_voice_and_wait("calm")
    celune.say("Speaking with non-default voice.")

    -- Call this or you'll have a bad engine state.
    celune.wait_until_idle()
    celune.sleep(1)

    -- Play a sound through Celune's pipeline.
    -- It can play formats supported by libsndfile.
    celune.play("extensions/NOT_TTS.wav")
    celune.sleep(1)

    -- Say something and don't save the output.
    celune.say("You will only hear this once.", false)
end)

-- Run actions on event dispatch.
subscribe("voice_changed", function(event)
    local old = event.old_voice
    local new = event.new_voice

    celune.log("Voice changed from "..old.." to "..new..".")
end)

-- This will be executed when you run "/invoke <your extension name>"."
function invoke(...)
    celune.log("You invoked the extension.")
    celune.say("You invoked the extension.")
end
