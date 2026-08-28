# SPDX-License-Identifier: Apache-2.0
"""Tests for Celune's CPU compatibility diagnostics."""

from unittest import mock

from celune.cpu import check_cpu_features, required_cpu_features


class TestCpuFeatures:
    """Verify CPU requirements are selected for each supported platform."""

    def test_x86_64_requires_avx_and_numpy_baseline(self) -> None:
        """Verify x86-64 uses one AVX baseline on every supported OS."""
        assert required_cpu_features("AMD64") == (
            "sse3",
            "ssse3",
            "sse4.1",
            "sse4.2",
            "popcnt",
            "cx16",
            "lahf_lm",
            "avx",
        )

    def test_linux_matches_the_x86_64_avx_baseline(self) -> None:
        """Verify Linux uses the same AVX baseline as Windows."""
        assert required_cpu_features("x86_64") == required_cpu_features("AMD64")

    def test_arm64_does_not_require_x86_features(self) -> None:
        """Verify ARM64 does not inherit x86 SIMD requirements."""
        assert not required_cpu_features("aarch64")

    def test_missing_features_make_check_unsupported(self) -> None:
        """Verify a detectable missing feature fails the compatibility check."""
        with (
            mock.patch("celune.cpu.platform.system", return_value="Windows"),
            mock.patch("celune.cpu.platform.machine", return_value="AMD64"),
            mock.patch(
                "celune.cpu._detect_cpu_features",
                return_value=frozenset({"sse3"}),
            ),
        ):
            result = check_cpu_features()

        assert not result.supported
        assert "sse4.2" in result.missing
