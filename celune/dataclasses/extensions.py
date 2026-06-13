"""Extension-facing dataclasses."""

from dataclasses import dataclass, field

from .. import __version__
from ..typing.common import JSONSerializable
from ..typing.extensions import (
    DevLogCallable,
    GetStateCallable,
    LogCallable,
    PlayCallable,
    SayCallable,
    SetVoiceCallable,
    StatusCallable,
    ThinkCallable,
    WaitUntilReadyCallable,
)

CELUNE_VERSION = __version__


@dataclass(slots=True)
class CeluneContext:
    """Celune's extension context."""

    log: LogCallable
    log_dev: DevLogCallable
    say: SayCallable
    think: ThinkCallable
    play: PlayCallable
    status: StatusCallable
    set_voice: SetVoiceCallable
    get_state: GetStateCallable
    wait_until_ready: WaitUntilReadyCallable
    name: str = "Celune"
    version: str = CELUNE_VERSION
    shared: dict[str, JSONSerializable] = field(default_factory=dict)
    dev: bool = False

    def expose(self, key: str, value: JSONSerializable) -> None:
        """Expose a shared object.

        Args:
            key: Shared-object name to publish.
            value: JSON-serializable value exposed to extensions.
        """
        self.shared[key] = value

    def get(
        self,
        key: str,
        default: JSONSerializable = None,
    ) -> JSONSerializable:
        """Get a shared object.

        Args:
            key: Shared-object name to read.
            default: Fallback returned when the key is missing.

        Returns:
            JSONSerializable: The stored value, or ``default`` when absent.
        """
        return self.shared.get(key, default)
