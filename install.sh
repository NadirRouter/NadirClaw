#!/bin/sh
# NadirClaw installer
# Usage: curl -fsSL https://raw.githubusercontent.com/NadirRouter/NadirClaw/main/install.sh | sh
set -e

REPO="https://github.com/NadirRouter/NadirClaw.git"
INSTALL_DIR="${NADIRCLAW_INSTALL_DIR:-$HOME/.nadirclaw}"
BIN_DIR="${NADIRCLAW_BIN_DIR:-/usr/local/bin}"

# ── Helpers ──────────────────────────────────────────────────

info()  { printf '\033[1;34m[nadirclaw]\033[0m %s\n' "$1"; }
ok()    { printf '\033[1;32m[nadirclaw]\033[0m %s\n' "$1"; }
err()   { printf '\033[1;31m[nadirclaw]\033[0m %s\n' "$1" >&2; }

command_exists() { command -v "$1" >/dev/null 2>&1; }

# ── Preflight ────────────────────────────────────────────────

info "Installing NadirClaw..."

# Check Python
PYTHON=""
if command_exists python3; then
    PYTHON="python3"
elif command_exists python; then
    PYTHON="python"
fi

if [ -z "$PYTHON" ]; then
    err "Python 3.10+ is required but not found."
    err "Install Python: https://www.python.org/downloads/"
    exit 1
fi

# Verify Python version >= 3.10
PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$($PYTHON -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$($PYTHON -c 'import sys; print(sys.version_info.minor)')

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    err "Python 3.10+ is required, found $PY_VERSION"
    exit 1
fi

info "Found Python $PY_VERSION"

# Check git
if ! command_exists git; then
    err "git is required but not found."
    exit 1
fi

# ── Install ──────────────────────────────────────────────────

# Clone or update
if [ -d "$INSTALL_DIR/.git" ]; then
    info "Updating existing installation at $INSTALL_DIR..."
    cd "$INSTALL_DIR"
    git pull --quiet origin main 2>/dev/null || git pull --quiet
elif [ -d "$INSTALL_DIR" ]; then
    # Directory exists but is not a git repo (e.g. created by credentials/logs).
    # Preserve user data, clone into a temp dir, then merge.
    info "Found $INSTALL_DIR (not a git repo). Installing into it..."
    TMPDIR_CLONE="$(mktemp -d)"
    git clone --quiet --depth 1 "$REPO" "$TMPDIR_CLONE"
    # Move git history and source files in, but don't overwrite user data
    cp -rn "$TMPDIR_CLONE/." "$INSTALL_DIR/" 2>/dev/null || true
    # Ensure .git and source files are present
    cp -r "$TMPDIR_CLONE/.git" "$INSTALL_DIR/.git"
    cp -r "$TMPDIR_CLONE/nadirclaw" "$INSTALL_DIR/nadirclaw"
    cp "$TMPDIR_CLONE/pyproject.toml" "$INSTALL_DIR/pyproject.toml"
    cp "$TMPDIR_CLONE/install.sh" "$INSTALL_DIR/install.sh" 2>/dev/null || true
    rm -rf "$TMPDIR_CLONE"
    cd "$INSTALL_DIR"
else
    info "Cloning NadirClaw to $INSTALL_DIR..."
    git clone --quiet --depth 1 "$REPO" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Create venv
if [ ! -d "$INSTALL_DIR/venv" ]; then
    info "Creating virtual environment..."
    $PYTHON -m venv "$INSTALL_DIR/venv"
fi

# Install package
info "Installing dependencies (this may take a minute)..."
"$INSTALL_DIR/venv/bin/pip" install --quiet --upgrade pip
"$INSTALL_DIR/venv/bin/pip" install --quiet -e "$INSTALL_DIR"

# ── Create CLI wrapper ───────────────────────────────────────

WRAPPER="$INSTALL_DIR/bin/nadirclaw"
mkdir -p "$INSTALL_DIR/bin"

cat > "$WRAPPER" <<SCRIPT
#!/bin/sh
exec "$INSTALL_DIR/venv/bin/nadirclaw" "\$@"
SCRIPT
chmod +x "$WRAPPER"

# ── Symlink to PATH ──────────────────────────────────────────

NEEDS_PATH=false

# Try /usr/local/bin first (may need sudo)
if [ -w "$BIN_DIR" ]; then
    ln -sf "$WRAPPER" "$BIN_DIR/nadirclaw"
    info "Linked nadirclaw to $BIN_DIR/nadirclaw"
elif [ "$(id -u)" -eq 0 ]; then
    ln -sf "$WRAPPER" "$BIN_DIR/nadirclaw"
    info "Linked nadirclaw to $BIN_DIR/nadirclaw"
else
    # Try with sudo
    if command_exists sudo; then
        info "Linking to $BIN_DIR (requires sudo)..."
        if sudo ln -sf "$WRAPPER" "$BIN_DIR/nadirclaw" 2>/dev/null; then
            info "Linked nadirclaw to $BIN_DIR/nadirclaw"
        else
            NEEDS_PATH=true
        fi
    else
        NEEDS_PATH=true
    fi
fi

# ── Shell config (fallback if /usr/local/bin didn't work) ────

if [ "$NEEDS_PATH" = true ]; then
    info "Could not write to $BIN_DIR. Adding to shell PATH instead..."
    PATH_LINE="export PATH=\"$INSTALL_DIR/bin:\$PATH\""

    add_to_shell() {
        if [ -f "$1" ] && grep -qF "$INSTALL_DIR/bin" "$1" 2>/dev/null; then
            return 0
        fi
        if [ -f "$1" ] || [ "$2" = "create" ]; then
            printf '\n# NadirClaw\n%s\n' "$PATH_LINE" >> "$1"
            info "Added to $1"
        fi
    }

    SHELL_NAME=$(basename "${SHELL:-/bin/sh}")
    case "$SHELL_NAME" in
        zsh)  add_to_shell "$HOME/.zshrc" ;;
        bash)
            if [ "$(uname)" = "Darwin" ]; then
                add_to_shell "$HOME/.bash_profile"
            else
                add_to_shell "$HOME/.bashrc"
            fi
            ;;
        fish)
            mkdir -p "$HOME/.config/fish"
            FISH_LINE="set -gx PATH $INSTALL_DIR/bin \$PATH"
            if ! grep -qF "$INSTALL_DIR/bin" "$HOME/.config/fish/config.fish" 2>/dev/null; then
                printf '\n# NadirClaw\n%s\n' "$FISH_LINE" >> "$HOME/.config/fish/config.fish"
                info "Added to ~/.config/fish/config.fish"
            fi
            ;;
        *)    add_to_shell "$HOME/.profile" ;;
    esac

    export PATH="$INSTALL_DIR/bin:$PATH"
fi

# ── Done ─────────────────────────────────────────────────────

echo ""
ok "NadirClaw installed successfully!"
echo ""
echo "  Get started:"
echo "    nadirclaw serve --verbose          # start the router"
echo "    nadirclaw classify \"hello world\"   # test classification"
echo "    nadirclaw status                   # check config"
echo ""
echo "  Integrations:"
echo "    nadirclaw openclaw onboard         # configure OpenClaw"
echo "    nadirclaw codex onboard            # configure Codex"
echo ""
echo "  Configure models (optional):"
echo "    export NADIRCLAW_SIMPLE_MODEL=ollama/llama3.1:8b"
echo "    export NADIRCLAW_COMPLEX_MODEL=claude-sonnet-4-20250514"
echo "    export ANTHROPIC_API_KEY=sk-ant-..."
echo ""

if [ "$NEEDS_PATH" = true ]; then
    echo "  NOTE: Restart your shell or run:"
    echo "    source ~/.$(basename ${SHELL:-sh})rc"
    echo ""
fi
