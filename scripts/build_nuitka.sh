#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if (( $# > 1 )); then
    echo "Usage: scripts/build_nuitka.sh [signed]" >&2
    exit 2
fi
build_mode="${1:-unsigned}"
if [[ "$build_mode" != "unsigned" && "$build_mode" != "signed" ]]; then
    echo "Build mode must be 'unsigned' or 'signed'." >&2
    exit 2
fi

signing_key_file="${CELUNE_RELEASE_SIGNING_KEY_FILE:-}"
require_signature="${CELUNE_REQUIRE_RELEASE_SIGNATURE:-0}"
if [[ "$build_mode" == "signed" ]]; then
    require_signature=1
fi
if [[ "$require_signature" == "1" ]]; then
    signing_key_file="${signing_key_file:-${HOME:?HOME must be set for signed builds}/.celune/release-signing-key.pem}"
fi
if [[ -n "$signing_key_file" && ! -f "$signing_key_file" ]]; then
    echo "The release signing key file was not found: $signing_key_file" >&2
    exit 1
fi
if [[ "$require_signature" == "1" && -z "$signing_key_file" ]]; then
    echo "A release signing key is required when signature enforcement is enabled." >&2
    exit 1
fi

export CFLAGS="-O2 -DNDEBUG -fstack-protector-strong -D_FORTIFY_SOURCE=3"
export CXXFLAGS="$CFLAGS"
export LDFLAGS="-Wl,-z,relro,-z,now"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$repo_root/bin"
build_python="3.13"
archive_name="Celune-linux-x64.zip"
if [[ "$build_mode" == "signed" || -n "$signing_key_file" ]]; then
    archive_name="Celune-linux-x64-signed.zip"
fi
archive_path="$output_dir/$archive_name"
manifest_script="$repo_root/scripts/write_update_manifest.py"
app_dir="$output_dir/Celune.AppDir"
desktop_src="$repo_root/Celune.AppDir/celune.desktop"
icon_src="$repo_root/Celune.AppDir/celune.png"

export UV_CACHE_DIR="$repo_root/.uv-cache"
if [[ "$repo_root" == /mnt/* ]]; then
    export UV_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/celune-uv"
fi

revision="$(python3 "$repo_root/scripts/root.py" --revision)"

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
    "$repo_root/launcher/signature.c"
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

if ! command -v pkg-config >/dev/null 2>&1 || ! pkg-config --exists libsodium; then
    echo "libsodium development files and pkg-config are required to build the Linux launcher." >&2
    exit 1
fi

sodium_libdir="$(pkg-config --variable=libdir libsodium)"
sodium_archive="$sodium_libdir/libsodium.a"
if [[ ! -f "$sodium_archive" ]]; then
    echo "The static libsodium archive was not found: $sodium_archive" >&2
    exit 1
fi
read -r -a sodium_cflags <<< "$(pkg-config --cflags libsodium)"
read -r -a sodium_private_flags <<< "$(pkg-config --static --libs-only-other libsodium)"

if ! command -v appimagetool >/dev/null 2>&1; then
    echo "appimagetool is required to create the AppImage." >&2
    exit 1
fi

if ! command -v zip >/dev/null 2>&1; then
    echo "zip is required to create the Linux release archive." >&2
    exit 1
fi

mkdir -p "$UV_CACHE_DIR"

signature_test="$(mktemp)"
trap 'rm -f "$signature_test"' EXIT
gcc -O2 -Wall -Wextra -Wpedantic \
	"${sodium_cflags[@]}" -I"$repo_root/launcher" \
	-o "$signature_test" \
	"$repo_root/tests/launcher_signature_test.c" \
	"$repo_root/launcher/signature.c" \
	"$sodium_archive" "${sodium_private_flags[@]}"
(cd "$repo_root" && "$signature_test")
rm -f "$signature_test"
trap - EXIT

python3 "$repo_root/scripts/root.py"

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
	-flto "${sodium_cflags[@]}" -I"$repo_root/launcher" \
	-Wl,-z,relro,-z,now -o "$output_dir/celune" \
	"${launcher_sources[@]}" "$sodium_archive" "${sodium_private_flags[@]}"
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

manifest_arguments=(
    --output-dir "$output_dir"
    --version "$version"
    --revision "$revision"
    --artifact "Celune-linux-$appimage_arch"
    --file celune
    --file celune-bin
    --file celune.AppImage
)
if [[ -n "$signing_key_file" ]]; then
    manifest_arguments+=(--signing-key "$signing_key_file")
fi
if [[ "$require_signature" == "1" ]]; then
    manifest_arguments+=(--require-signature)
fi

PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" uv run python "$manifest_script" \
    "${manifest_arguments[@]}"

(
    cd "$output_dir"
    archive_files=(
        celune \
        celune-bin \
        celune.AppImage \
        celune-update.json
    )
    if [[ -f celune-update.sig ]]; then
        archive_files+=(celune-update.sig)
    fi
    zip -q -9 -j "$archive_path" "${archive_files[@]}"
)
