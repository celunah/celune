# SPDX-License-Identifier: Apache-2.0
"""Helpers for assembling source-size-limited modules."""

import inspect


def install_class_functions(target, functions, *, properties=()):
    """Install explicitly imported methods on an existing public class."""
    for name, value in functions.items():
        if name in properties:
            value = property(value)
        elif inspect.isfunction(value) and next(
            iter(inspect.signature(value).parameters), None
        ) not in {"self", "engine"}:
            value = staticmethod(value)
        setattr(target, name, value)


def install_module_functions(namespace, functions):
    """Install explicitly imported functions in a module namespace."""
    for name, value in functions.items():
        namespace[name] = value
