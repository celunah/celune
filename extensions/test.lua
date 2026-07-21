EXTENSION_NAME = "Test"

subscribe("ready", function(event)
    celune.log("Celune Lua runtime is running this extension on ".._VERSION)
    celune.sleep(5)
    celune.log("Log test")
    celune.sleep(1)
    celune.status("Status test")
    celune.sleep(5)
    celune.status("Status test (warning)", "warning")
    celune.sleep(5)
    celune.status("Status test (error)", "error")
    celune.sleep(5)
    celune.status("Status test (unknown)", "invalid")
    celune.sleep(5)
    celune.say("Speaking with default voice.")
    celune.sleep(1)
    celune.set_voice_and_wait("calm")
    celune.say("Speaking with non-default voice.")
    celune.wait_until_idle()
    celune.sleep(1)
    celune.play("extensions/NOT_TTS.wav")
    celune.sleep(1)
    celune.say("You will only hear this once.", false)
end)

subscribe("voice_changed", function(event)
    local old = event.old_voice
    local new = event.new_voice

    celune.log("Voice changed from "..old.." to "..new..".")
end)

function invoke(...)
    celune.log("You invoked the extension.")
    celune.say("You invoked the extension.")
end
