#!/data/data/com.termux/files/usr/bin/bash
# =========================================================================
#  AI Voice — Local STT + LLM + TTS for Termux / Linux
#  Background daemon + thin client architecture
#  Whisper (STT) → llama.cpp/Ollama (LLM) → Piper / termux-tts (TTS)
#  Version: 2026-10-17 (background layer)
# =========================================================================

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

INSTALL_DIR="${HOME}/ai-voice"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Early subcommands (before any install work) ───────────────────────────
case "${1:-}" in
    run)
        if [ ! -f "$INSTALL_DIR/voice_loop.py" ] && [ ! -f "$INSTALL_DIR/voice-loop.py" ]; then
            echo -e "${RED}Not installed. Run: bash $0${NC}"
            exit 1
        fi
        LOOP="$INSTALL_DIR/voice_loop.py"
        [ -f "$LOOP" ] || LOOP="$INSTALL_DIR/voice-loop.py"
        if ! python3 "$INSTALL_DIR/voice_client.py" status >/dev/null 2>&1; then
            echo -e "${YELLOW}Daemon not running — starting it…${NC}"
            "$0" daemon
            sleep 1.5
        fi
        exec python3 "$LOOP"
        ;;
    daemon)
        DAEMON="$INSTALL_DIR/voice_daemon.py"
        [ -f "$DAEMON" ] || DAEMON="$INSTALL_DIR/voice-daemon.py"
        if [ ! -f "$DAEMON" ]; then
            echo -e "${RED}Daemon not found. Run installer first: bash $0${NC}"
            exit 1
        fi
        if python3 "$INSTALL_DIR/voice_client.py" status >/dev/null 2>&1; then
            echo -e "${YELLOW}Daemon already running.${NC}"
            python3 "$INSTALL_DIR/voice_client.py" status
            exit 0
        fi
        mkdir -p "$INSTALL_DIR/logs"
        nohup python3 "$DAEMON" >> "$INSTALL_DIR/logs/daemon.out" 2>&1 &
        echo "daemon started (pid $!)"
        sleep 0.5
        python3 "$INSTALL_DIR/voice_client.py" status 2>/dev/null || true
        exit 0
        ;;
    stop)
        if [ -f "$INSTALL_DIR/voice_client.py" ]; then
            python3 "$INSTALL_DIR/voice_client.py" stop 2>/dev/null || true
        fi
        if [ -f "$INSTALL_DIR/daemon.pid" ]; then
            kill "$(cat "$INSTALL_DIR/daemon.pid")" 2>/dev/null || true
            rm -f "$INSTALL_DIR/daemon.pid" "$INSTALL_DIR/ctl.sock"
        fi
        echo "daemon stopped"
        exit 0
        ;;
    status)
        if [ -f "$INSTALL_DIR/voice_client.py" ]; then
            python3 "$INSTALL_DIR/voice_client.py" status
        else
            echo '{"ok":false,"error":"not_installed"}'
        fi
        exit 0
        ;;
    help|--help|-h)
        echo "Usage:"
        echo "  bash $0            # install / re-install"
        echo "  bash $0 daemon     # start background daemon"
        echo "  bash $0 run        # start voice loop (auto-starts daemon)"
        echo "  bash $0 status     # daemon status"
        echo "  bash $0 stop       # stop daemon"
        echo "  bash $0 help"
        echo ""
        echo "Prerequisites:"
        echo "  • llama-server on :8080  OR  ollama serve"
        echo "  • Termux:API app + pkg install termux-api (mic + TTS)"
        echo "  • Mic permission granted to Termux"
        echo "  • Battery unrestricted for Termux (Android Settings)"
        exit 0
        ;;
esac

# ── Installer path (no args or unknown) ───────────────────────────────────
echo -e "${CYAN}${BOLD}"
cat << 'EOF'
   █████╗ ██╗    ██╗   ██╗ ██████╗ ██╗ ██████╗███████╗
  ██╔══██╗██║    ██║   ██║██╔═══██╗██║██╔════╝██╔════╝
  ███████║██║    ██║   ██║██║   ██║██║██║     █████╗
  ██╔══██║██║    ╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝
  ██║  ██║██║     ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗
  ╚═╝  ╚═╝╚═╝      ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝
EOF
echo -e "${NC}${BOLD}  Local Voice AI — STT + LLM + TTS (daemon edition)${NC}\n"

IS_TERMUX=false
[[ -n "${TERMUX_VERSION:-}" || "${PREFIX:-}" == *"termux"* ]] && IS_TERMUX=true

mkdir -p "$INSTALL_DIR"/{voices,models,logs}
cd "$INSTALL_DIR"

# ── 1. System deps ────────────────────────────────────────────────────────
echo -e "${YELLOW}[1/5] Installing system packages...${NC}"
if $IS_TERMUX; then
    pkg update -y
    pkg install -y python clang git wget ffmpeg sox termux-api
    pkg install -y portaudio 2>/dev/null || true
else
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -y
        sudo apt-get install -y python3 python3-pip git wget ffmpeg sox \
            portaudio19-dev python3-dev build-essential
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y python3 python3-pip git wget ffmpeg sox portaudio-devel
    else
        echo -e "${YELLOW}Install python3, pip, ffmpeg, sox, portaudio manually if needed.${NC}"
    fi
fi

# ── 2. Python deps ────────────────────────────────────────────────────────
echo -e "${YELLOW}[2/5] Installing Python packages...${NC}"
PIP_FLAGS=""
$IS_TERMUX && PIP_FLAGS="--break-system-packages" || PIP_FLAGS="--user"

python3 -m pip install $PIP_FLAGS --quiet --upgrade pip setuptools wheel
python3 -m pip install $PIP_FLAGS --quiet requests rich soundfile numpy

echo -e "${YELLOW}  Installing Whisper + Piper (may take a few minutes)...${NC}"
python3 -m pip install $PIP_FLAGS --quiet openai-whisper piper-tts 2>/dev/null || {
    echo -e "${YELLOW}Warning: Whisper/Piper install failed — Termux TTS fallback still works.${NC}"
}
python3 -m pip install $PIP_FLAGS --quiet sounddevice 2>/dev/null || true

# ── 3. Piper voice ────────────────────────────────────────────────────────
echo -e "${YELLOW}[3/5] Downloading Piper voice (en_US-lessac-medium)...${NC}"
PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
VOICE_DIR="$INSTALL_DIR/voices"
if [ ! -f "$VOICE_DIR/en_US-lessac-medium.onnx" ]; then
    wget -q --show-progress -O "$VOICE_DIR/en_US-lessac-medium.onnx" \
        "$PIPER_BASE/en_US-lessac-medium.onnx" 2>/dev/null || \
        echo -e "${YELLOW}Voice download deferred — will use termux-tts until available.${NC}"
    wget -q -O "$VOICE_DIR/en_US-lessac-medium.onnx.json" \
        "$PIPER_BASE/en_US-lessac-medium.onnx.json" 2>/dev/null || true
fi

# ── 4. Config ─────────────────────────────────────────────────────────────
echo -e "${YELLOW}[4/5] Writing configuration...${NC}"
cat > "$INSTALL_DIR/config.json" << 'EOF'
{
  "whisper_model": "base",
  "whisper_language": "en",
  "sample_rate": 16000,
  "record_seconds": 5,
  "use_termux_mic": true,
  "llm": {
    "backend": "llamacpp",
    "url": "http://127.0.0.1:8080/v1/chat/completions",
    "model": "local",
    "system_prompt": "You are a helpful, concise voice assistant. Keep answers under 3 sentences unless asked for detail.",
    "max_tokens": 200,
    "temperature": 0.7
  },
  "tts": {
    "voice": "en_US-lessac-medium",
    "speed": 1.0,
    "prefer_termux_tts": false
  }
}
EOF

# ── 5. Install Python modules from package ────────────────────────────────
echo -e "${YELLOW}[5/5] Installing daemon + client + loop...${NC}"

copy_or_fail() {
    local src="$1" dest="$2"
    if [ -f "$src" ]; then
        cp -f "$src" "$dest"
        chmod +x "$dest"
        return 0
    fi
    return 1
}

OK=0
copy_or_fail "$SCRIPT_DIR/voice_daemon.py" "$INSTALL_DIR/voice_daemon.py" && OK=1
copy_or_fail "$SCRIPT_DIR/voice_client.py" "$INSTALL_DIR/voice_client.py" && OK=1
copy_or_fail "$SCRIPT_DIR/voice_loop.py"   "$INSTALL_DIR/voice_loop.py"   && OK=1
copy_or_fail "$SCRIPT_DIR/minimal-voice.py" "$INSTALL_DIR/minimal-voice.py" || true

cp -f "$SCRIPT_DIR/ai-voice.sh" "$INSTALL_DIR/ai-voice.sh" 2>/dev/null || true
chmod +x "$INSTALL_DIR/ai-voice.sh" 2>/dev/null || true

if [ "$OK" -eq 0 ]; then
    echo -e "${RED}Could not find voice_daemon.py / voice_client.py / voice_loop.py next to installer.${NC}"
    echo -e "${YELLOW}Place the full package in one folder and re-run bash ai-voice.sh${NC}"
    exit 1
fi

ln -sf voice_daemon.py "$INSTALL_DIR/voice-daemon.py" 2>/dev/null || true
ln -sf voice_client.py "$INSTALL_DIR/voice-client.py" 2>/dev/null || true
ln -sf voice_loop.py   "$INSTALL_DIR/voice-loop.py"   2>/dev/null || true

echo -e "${GREEN}${BOLD}"
echo "╔════════════════════════════════════════════════╗"
echo "║   AI Voice installed! 🎙️                       ║"
echo "║                                                ║"
echo "║   1. Start LLM:  bash ~/start-ai.sh            ║"
echo "║      (or: ollama serve &)                      ║"
echo "║                                                ║"
echo "║   2. Start daemon:                             ║"
echo "║      bash ~/ai-voice/ai-voice.sh daemon        ║"
echo "║                                                ║"
echo "║   3. Voice loop:                               ║"
echo "║      bash ~/ai-voice/ai-voice.sh run           ║"
echo "║                                                ║"
echo "║   status / stop:                               ║"
echo "║      bash ~/ai-voice/ai-voice.sh status|stop   ║"
echo "╚════════════════════════════════════════════════╝"
echo -e "${NC}"

if $IS_TERMUX; then
    echo -e "${YELLOW}Tip: Disable battery optimization for Termux${NC}"
    echo -e "  Android Settings → Apps → Termux → Battery → Unrestricted"
    echo ""
fi
