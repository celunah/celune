#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

export CFLAGS="-O2 -DNDEBUG -fstack-protector-strong -D_FORTIFY_SOURCE=3"
export CXXFLAGS="$CFLAGS"
export LDFLAGS="-Wl,-z,relro,-z,now"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$repo_root/bin"
build_python="3.13"
archive_path="$output_dir/Celune-linux-x64.zip"
manifest_script="$repo_root/scripts/write_update_manifest.py"
app_dir="$output_dir/Celune.AppDir"
desktop_src="$repo_root/Celune.AppDir/celune.desktop"
icon_src="$repo_root/Celune.AppDir/celune.png"

if pgrep -x celune >/dev/null || pgrep -x celune-bin >/dev/null; then
    echo "Celune is already running, terminating before proceeding with build."
    pkill -TERM -x celune || true
    pkill -TERM -x celune-bin || true
    sleep 1
    pkill -KILL -x celune || true
    pkill -KILL -x celune-bin || true
fi

version_line="$(grep -m1 '^version = "' "$repo_root/pyproject.toml")"
if [[ -z "$version_line" ]]; then
    echo "Could not determine the project version from pyproject.toml." >&2
    exit 1
fi

version="${version_line#version = \"}"
version="${version%\"}"

if [[ ! -f "$repo_root/nuitka_main.py" ]]; then
    echo "nuitka_main.py was not found." >&2
    exit 1
fi

if [[ ! -f "$manifest_script" ]]; then
    echo "The update manifest script was not found." >&2
    exit 1
fi

launcher_sources=(
    "$repo_root/launcher/launcher.c"
    "$repo_root/launcher/unix/runtime.c"
    "$repo_root/launcher/unix/terminal.c"
)
for launcher_source in "${launcher_sources[@]}"; do
    if [[ ! -f "$launcher_source" ]]; then
        echo "Launcher source was not found: $launcher_source" >&2
        exit 1
    fi
done

if [[ ! -f "$desktop_src" || ! -f "$icon_src" ]]; then
    echo "Celune.AppDir metadata files were not found." >&2
    exit 1
fi

if ! command -v gcc >/dev/null 2>&1; then
    echo "gcc is required to build the Linux launcher." >&2
    exit 1
fi

if ! command -v appimagetool >/dev/null 2>&1; then
    echo "appimagetool is required to create the AppImage." >&2
    exit 1
fi

if ! command -v zip >/dev/null 2>&1; then
    echo "zip is required to create the Linux release archive." >&2
    exit 1
fi

export UV_CACHE_DIR="$repo_root/.uv-cache"
if [[ "$repo_root" == /mnt/* ]]; then
    export UV_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/celune-uv"
fi
mkdir -p "$UV_CACHE_DIR"

mkdir -p "$output_dir"
rm -rf \
    "$archive_path" \
    "$output_dir/default_config.yaml" \
    "$output_dir/voices" \
    "$output_dir/resources" \
    "$output_dir/assets"

uv run --python "$build_python" python -m nuitka \
    --deployment \
    --follow-import-to=celune \
    --include-package-data=celune \
    --output-dir="$output_dir" \
    --output-filename=celune-bin \
    --lto=yes \
    "$repo_root/nuitka_main.py"

gcc -O2 -s -Wall -Wextra -Wpedantic \
	-DNDEBUG -D_FORTIFY_SOURCE=3 -fstack-protector-strong \
	-flto -I"$repo_root/launcher" -Wl,-z,relro,-z,now -o "$output_dir/celune" "${launcher_sources[@]}"
chmod +x "$output_dir/celune" "$output_dir/celune-bin"

rm -rf "$output_dir/nuitka_main.build"

rm -rf "$app_dir"
mkdir -p "$app_dir"
cp "$desktop_src" "$app_dir/celune.desktop"
cp "$icon_src" "$app_dir/celune.png"
ln -sfn "celune.png" "$app_dir/.DirIcon"
cat > "$app_dir/AppRun" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

appimage_path="${APPIMAGE:-$0}"
appimage_dir="$(cd "$(dirname "$(readlink -f "$appimage_path")")" && pwd)"
launcher="$appimage_dir/celune"

if [[ ! -x "$launcher" ]]; then
    echo "Celune launcher not found beside AppImage: $launcher" >&2
    exit 1
fi

exec "$launcher" "$@"
EOF
chmod +x "$app_dir/AppRun"

arch="${ARCH:-$(uname -m)}"
case "$arch" in
    x86_64|amd64)
        appimage_arch="x86_64"
        ;;
    aarch64|arm64)
        appimage_arch="aarch64"
        ;;
    *)
        appimage_arch="$arch"
        ;;
esac

ARCH="$appimage_arch" appimagetool "$app_dir" "$output_dir/celune.AppImage"
rm -rf "$app_dir"

revision="${CELUNE_REVISION:-$(git -C "$repo_root" rev-parse HEAD)}"
if [[ -z "$revision" ]]; then
    echo "Could not determine the Git revision for update metadata." >&2
    exit 1
fi

uv run python "$manifest_script" \
    --output-dir "$output_dir" \
    --version "$version" \
    --revision "$revision" \
    --artifact "Celune-linux-$appimage_arch" \
    --file celune \
    --file celune-bin \
    --file celune.AppImage

(
    cd "$output_dir"
    zip -q -9 -j "$archive_path" \
        celune \
        celune-bin \
        celune.AppImage \
        celune-update.json
)
