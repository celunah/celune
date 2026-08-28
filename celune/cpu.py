# SPDX-License-Identifier: Apache-2.0
"""Detect CPU instruction sets required by Celune's native dependencies."""

import ctypes
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = ["CpuFeatureCheck", "check_cpu_features", "required_cpu_features"]

_X86_64_MACHINES = frozenset({"amd64", "x64", "x86_64"})
_FEATURE_ORDER = (
    "sse3",
    "ssse3",
    "sse4.1",
    "sse4.2",
    "popcnt",
    "cx16",
    "lahf_lm",
    "avx",
    "avx2",
    "avx512f",
)
_X86_64_V2_FEATURES = frozenset(
    {"sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "cx16", "lahf_lm"}
)
_WINDOWS_CPUID = bytes(
    (
        0x53,
        0x89,
        0xC8,
        0x89,
        0xD1,
        0x0F,
        0xA2,
        0x41,
        0x89,
        0x00,
        0x41,
        0x89,
        0x58,
        0x04,
        0x41,
        0x89,
        0x48,
        0x08,
        0x41,
        0x89,
        0x50,
        0x0C,
        0x5B,
        0xC3,
    )
)


@dataclass(frozen=True)
class CpuFeatureCheck:
    """Describe the CPU features available to the current process."""

    architecture: str
    required: tuple[str, ...]
    available: frozenset[str]
    detectable: bool

    @property
    def missing(self) -> tuple[str, ...]:
        """Return required features that were not detected."""
        return tuple(
            feature for feature in self.required if feature not in self.available
        )

    @property
    def supported(self) -> bool:
        """Return whether the current CPU satisfies Celune's requirements."""
        return self.detectable and not self.missing


def _canonical_architecture(machine: str) -> str:
    """Return the architecture name used by the CPU requirement table."""
    normalized = machine.strip().lower()
    if normalized in _X86_64_MACHINES:
        return "x86_64"
    if normalized in {"aarch64", "arm64"}:
        return "arm64"
    return normalized


def required_cpu_features(machine: Optional[str] = None) -> tuple[str, ...]:
    """Return the instruction sets required by the active Celune platform.

    Args:
        machine: Optional platform machine name for testing.

    Returns:
        tuple[str, ...]: Required canonical CPU feature names in display order.
    """
    resolved_machine = _canonical_architecture(machine or platform.machine())
    if resolved_machine != "x86_64":
        return ()

    # Pedalboard's native builds use AVX on x86. NumPy supplies the
    # x86-64-v2 baseline shared by Celune and its backend workers.
    required = set(_X86_64_V2_FEATURES)
    required.add("avx")
    return tuple(feature for feature in _FEATURE_ORDER if feature in required)


def _read_linux_cpu_features() -> Optional[frozenset[str]]:
    """Read normalized CPU flags from Linux's process-visible CPU report."""
    try:
        cpu_info = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return None

    aliases = {
        "pni": "sse3",
        "sse4_1": "sse4.1",
        "sse4_2": "sse4.2",
    }
    features: set[str] = set()
    for line in cpu_info.splitlines():
        if ":" not in line:
            continue
        label, values = line.split(":", 1)
        if label.strip().lower() not in {"flags", "features"}:
            continue
        features.update(aliases.get(value, value) for value in values.split())
    return frozenset(features)


def _windows_cpuid_features() -> Optional[frozenset[str]]:
    """Read x86 CPU feature bits through Windows' CPUID instruction."""
    if platform.machine().strip().lower() not in _X86_64_MACHINES:
        return frozenset()

    kernel32: Optional[ctypes.CDLL] = None
    address: Optional[int] = None
    try:
        kernel32 = ctypes.CDLL("kernel32", use_last_error=True)
        kernel32.VirtualAlloc.restype = ctypes.c_void_p
        kernel32.VirtualAlloc.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        kernel32.VirtualFree.restype = ctypes.c_bool
        kernel32.VirtualFree.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
        ]
        address = kernel32.VirtualAlloc(
            None,
            len(_WINDOWS_CPUID),
            0x3000,
            0x40,
        )
        if not address:
            return None
        ctypes.memmove(address, _WINDOWS_CPUID, len(_WINDOWS_CPUID))
        function_type = ctypes.CFUNCTYPE(
            None,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        )
        cpuid = function_type(address)
        registers = (ctypes.c_uint32 * 4)()

        def read_leaf(leaf: int, subleaf: int = 0) -> tuple[int, int, int, int]:
            cpuid(leaf, subleaf, registers)
            return (
                int(registers[0]),
                int(registers[1]),
                int(registers[2]),
                int(registers[3]),
            )

        max_basic, _, _, _ = read_leaf(0)
        _, _, leaf_one_ecx, _ = read_leaf(1) if max_basic >= 1 else (0, 0, 0, 0)
        max_extended, _, _, _ = read_leaf(0x80000000)
        _, _, extended_ecx, _ = (
            read_leaf(0x80000001) if max_extended >= 0x80000001 else (0, 0, 0, 0)
        )
        leaf_seven_ebx = read_leaf(7)[1] if max_basic >= 7 else 0

        features: set[str] = set()
        if leaf_one_ecx & (1 << 0):
            features.add("sse3")
        if leaf_one_ecx & (1 << 9):
            features.add("ssse3")
        if leaf_one_ecx & (1 << 19):
            features.add("sse4.1")
        if leaf_one_ecx & (1 << 20):
            features.add("sse4.2")
        if leaf_one_ecx & (1 << 23):
            features.add("popcnt")
        if leaf_one_ecx & (1 << 13):
            features.add("cx16")
        if leaf_one_ecx & (1 << 27) and leaf_one_ecx & (1 << 28):
            features.add("avx")
        if leaf_seven_ebx & (1 << 5):
            features.add("avx2")
        if leaf_seven_ebx & (1 << 16):
            features.add("avx512f")
        if extended_ecx & (1 << 0):
            features.add("lahf_lm")
        return frozenset(features)
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    finally:
        if kernel32 is not None and address:
            kernel32.VirtualFree(address, 0, 0x8000)


def _detect_cpu_features(
    system_name: str, architecture: str
) -> Optional[frozenset[str]]:
    """Detect CPU features using the operating system's native facilities."""
    if architecture != "x86_64":
        return frozenset()
    if system_name == "Linux":
        return _read_linux_cpu_features()
    if system_name == "Windows":
        return _windows_cpuid_features()
    return None


def check_cpu_features() -> CpuFeatureCheck:
    """Check the current CPU against Celune's native dependency baseline."""
    system_name = platform.system()
    architecture = _canonical_architecture(platform.machine())
    required = required_cpu_features(architecture)
    if not required:
        return CpuFeatureCheck(architecture, required, frozenset(), True)

    available = _detect_cpu_features(system_name, architecture)
    return CpuFeatureCheck(
        architecture,
        required,
        available or frozenset(),
        available is not None,
    )
