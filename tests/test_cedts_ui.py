"""Tests for CEDTS-framed frontend timed updates."""

from celune.cedts.ui import UiTimedUpdate, UiTimedUpdateChannel


def test_ui_timed_update_channel_delivers_framed_updates() -> None:
    """Verify a timed update survives the CEDTS event envelope."""
    channel = UiTimedUpdateChannel()
    received: list[UiTimedUpdate] = []
    unsubscribe = channel.subscribe(received.append)
    update = UiTimedUpdate(
        runtime_id="runtime",
        sequence=7,
        emitted_at=12.5,
        resource_page=3,
        theme_name="celune_light",
        status_text="Speaking",
        status_severity="info",
        status_marquee_offset=2,
    )

    channel.publish(update)
    unsubscribe()
    channel.publish(update)

    assert received == [update]
