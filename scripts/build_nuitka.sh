#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$repo_root/build/nuitka"

version_line="$(grep -m1 '^version = "' "$repo_root/pyproject.toml")"
if [[ -z "$version_line" ]]; then
    echo "Could not determine the project version from pyproject.toml." >&2
    exit 1
fi

version="${version_line#version = \"}"
version="${version%\"}"
windows_version="${version%%+*}"

if [[ ! "$windows_version" =~ ^[0-9]+(\.[0-9]+){0,3}$ ]]; then
    echo "The project version '$windows_version' is not a valid version string." >&2
    exit 1
fi

if [[ ! -f "$repo_root/nuitka_main.py" ]]; then
    echo "nuitka_main.py was not found." >&2
    exit 1
fi

if [[ ! -f "$repo_root/resources/celune.res" ]]; then
    echo "resources/celune.res was not found." >&2
    exit 1
fi

export UV_CACHE_DIR="$repo_root/.uv-cache"

uv run python -m nuitka \
    --deployment \
    --follow-import-to=celune \
    --include-package-data=celune \
    --include-data-files="$repo_root/default_config.yaml=default_config.yaml" \
    --include-data-dir="$repo_root/voices=voices" \
    --include-data-dir="$repo_root/resources=resources" \
    --product-name=Celune \
    --file-description=Celune \
    --product-version="$windows_version" \
    --file-version="$windows_version" \
    --output-dir="$output_dir" \
    --output-filename=celune \
    "$repo_root/nuitka_main.py"

mkdir -p "$output_dir/celune"
cp "$repo_root/default_config.yaml" "$output_dir/default_config.yaml"
cp -R "$repo_root/voices" "$output_dir/voices"
cp -R "$repo_root/resources" "$output_dir/resources"
cp -R "$repo_root/celune/assets" "$output_dir/celune/assets"
