#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return 0
    fi
    for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

UV_BIN=$(find_uv || true)
if [ -z "$UV_BIN" ]; then
    printf '%s\n' "LikenessForge: uv not found; installing Astral uv…"
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        printf '%s\n' "Unable to install uv automatically: curl or wget is required." >&2
        exit 1
    fi
    UV_BIN=$(find_uv || true)
fi

if [ -z "$UV_BIN" ]; then
    printf '%s\n' "uv installation completed but the executable could not be located." >&2
    exit 1
fi

printf '%s\n' "LikenessForge: using $($UV_BIN --version)"
printf '%s\n' "LikenessForge: ensuring the project Python from .python-version is installed…"
"$UV_BIN" python install
printf '%s\n' "LikenessForge: creating/updating the managed environment and GUI/model dependencies…"
"$UV_BIN" sync --extra all
printf '%s\n' "LikenessForge: starting GUI…"
exec "$UV_BIN" run --no-sync likenessforge-gui "$@"
