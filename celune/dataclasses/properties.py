# SPDX-License-Identifier: Apache-2.0
"""Property helpers for grouped Celune runtime state."""

from dataclasses import dataclass
from typing import Optional

from ..typing.aliases import ConstantPropertyValue


@dataclass(frozen=True, slots=True)
class ForwardedPropertySpec:
    """Describe one property forwarded into a state container."""

    name: str
    container_name: str
    field_name: str
    doc: Optional[str] = None
    read_only: bool = False


@dataclass(frozen=True, slots=True)
class ConstantPropertySpec:
    """Describe one constant-backed property."""

    name: str
    value: ConstantPropertyValue
    doc: Optional[str] = None


def forward_property(
    container_name: str,
    field_name: str,
    *,
    doc: Optional[str] = None,
    read_only: bool = False,
) -> property:
    """Create a property that forwards storage to a grouped state container.

    Args:
        container_name: Attribute holding the grouped state object.
        field_name: Field name inside that grouped state object.
        doc: Optional property docstring to attach.
        read_only: Whether the generated property should omit a setter.

    Returns:
        property: A descriptor that reads from the grouped state container.
    """

    def getter(instance):
        return getattr(getattr(instance, container_name), field_name)

    if read_only:
        return property(getter, doc=doc)

    def setter(instance, value) -> None:
        setattr(getattr(instance, container_name), field_name, value)

    return property(getter, setter, doc=doc)


def constant_property(
    value: ConstantPropertyValue, *, doc: Optional[str] = None
) -> property:
    """Create a read-only property that always returns one constant value.

    Args:
        value: Constant value returned by the property.
        doc: Optional property docstring to attach.

    Returns:
        property: A descriptor that always returns ``value``.
    """

    def getter(_instance):
        return value

    return property(getter, doc=doc)


def bind_forwarded_properties(
    namespace: dict[str, property],
    specs: tuple[ForwardedPropertySpec, ...],
) -> None:
    """Populate a class namespace with forwarded properties.

    Args:
        namespace: Class-body namespace being assembled.
        specs: Forwarding definitions to install into the namespace.
    """
    for spec in specs:
        namespace[spec.name] = forward_property(
            spec.container_name,
            spec.field_name,
            doc=spec.doc,
            read_only=spec.read_only,
        )


def bind_constant_properties(
    namespace: dict[str, property],
    specs: tuple[ConstantPropertySpec, ...],
) -> None:
    """Populate a class namespace with constant-backed properties.

    Args:
        namespace: Class-body namespace being assembled.
        specs: Constant-property definitions to install into the namespace.
    """
    for spec in specs:
        namespace[spec.name] = constant_property(spec.value, doc=spec.doc)
