#!/usr/bin/env bash
# install.sh — AOC v5 installer and launcher
# Usage: bash install.sh [run|daemon|stop|status|magic|magic-engines|help]
set -e

BANNER="==========================================================
 AOC v5 — offline voice, synthesis, HUSH bus
=========================================================="
echo "$BANNER"

INSTALL_DIR="$HOME/ai-voice"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Launcher dispatch ----
case "${1:-}" in
  run|voice)
    cd "$INSTALL_DIR" 2>/dev/null || { echo "Not installed. Run: bash install.sh"; exit 1; }
    if command -v aoc-voice >/dev/null 2>&1; then
      exec aoc-voice
    fi
    exec python3 -m hush_voice.loop
    ;;
  daemon)
    if command -v aoc-daemon >/dev/null 2>&1; then
      # if already running
      if python3 -c "from hush_voice.client import status; import sys; sys.exit(0 if status().get('ok') else 1)" 2>/dev/null; then
        echo "[daemon] already running"
        python3 -c "from hush_voice.client import status; import json; print(json.dumps(status()))"
        exit 0
      fi
      nohup aoc-daemon >> "$INSTALL_DIR/logs/daemon.out" 2>&1 &
      echo "[daemon] started (pid $!)"
      exit 0
    fi
    cd "$INSTALL_DIR" 2>/dev/null || { echo "Not installed."; exit 1; }
    mkdir -p logs
    nohup python3 -m hush_voice.daemon >> logs/daemon.out 2>&1 &
    echo "[daemon] started (pid $!)"
    exit 0
    ;;
  stop)
    python3 -c "from hush_voice.client import stop; print(stop())" 2>/dev/null || true
    rm -f "$INSTALL_DIR/daemon.pid" "$INSTALL_DIR/ctl.sock" 2>/dev/null || true
    echo "daemon stopped"
    exit 0
    ;;
  status)
    python3 -c "from hush_voice.client import status; import json; print(json.dumps(status()))" 2>/dev/null \
      || echo '{"ok": false, "error": "daemon_not_running"}'
    exit 0
    ;;
  magic)
    shift
    if command -v aoc-magic >/dev/null 2>&1; then
      exec aoc-magic "$@"
    fi
    exec python3 -m aoc_magic_v5 "$@"
    ;;
  magic-engines)
    if command -v aoc-magic-engines >/dev/null 2>&1; then
      exec aoc-magic-engines
    fi
    exec python3 -c "from aoc_magic_v5.cli import engines_main; raise SystemExit(engines_main())"
    ;;
  help)
    echo "Usage: bash $(basename "$0") [run|daemon|stop|status|magic|magic-engines|help]"
    echo "  (no args)  full install"
    exit 0
    ;;
esac

# ---- Install path ----
ON_TERMUX=0
if [ -d "/data/data/com.termux" ] || [ -n "${TERMUX_VERSION:-}" ] || [[ "${PREFIX:-}" == *termux* ]]; then
  ON_TERMUX=1
  echo "[*] Termux environment detected"
fi

mkdir -p "$INSTALL_DIR"/{voices/models,logs,models}
cd "$SRC_DIR"

# [1/6] System packages
echo "[1/6] System packages"
if [ "$ON_TERMUX" -eq 1 ]; then
  pkg update -y || true
  pkg install -y python ffmpeg libsndfile portaudio cmake clang git curl wget sox termux-api || true
else
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y || true
    sudo apt-get install -y python3 python3-pip ffmpeg libsndfile1 portaudio19-dev \
      cmake build-essential git curl wget sox || true
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y python3 python3-pip ffmpeg libsndfile portaudio-devel \
      cmake gcc-c++ git curl wget sox || true
  else
    echo "[!] Unknown distro — skipping system packages"
  fi
fi

# [2/6] Python packages
echo "[2/6] Python packages"
PIP_FLAGS=""
if [ "$ON_TERMUX" -eq 1 ]; then
  PIP_FLAGS="--break-system-packages"
else
  PIP_FLAGS="--user"
fi
python3 -m pip install $PIP_FLAGS --upgrade pip setuptools wheel || true
python3 -m pip install $PIP_FLAGS numpy soundfile requests rich || true
python3 -m pip install $PIP_FLAGS openai-whisper piper-tts pyttsx3 2>/dev/null || true
python3 -m pip install $PIP_FLAGS sounddevice 2>/dev/null || true

# [3/6] Piper voice
echo "[3/6] Piper voice"
VOICE_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
mkdir -p "$INSTALL_DIR/voices/models"
curl -L -o "$INSTALL_DIR/voices/models/en_US-lessac-medium.onnx" \
  "$VOICE_BASE/en_US-lessac-medium.onnx" 2>/dev/null || \
  wget -q -O "$INSTALL_DIR/voices/models/en_US-lessac-medium.onnx" \
  "$VOICE_BASE/en_US-lessac-medium.onnx" || true
curl -L -o "$INSTALL_DIR/voices/models/en_US-lessac-medium.onnx.json" \
  "$VOICE_BASE/en_US-lessac-medium.onnx.json" 2>/dev/null || \
  wget -q -O "$INSTALL_DIR/voices/models/en_US-lessac-medium.onnx.json" \
  "$VOICE_BASE/en_US-lessac-medium.onnx.json" || true
cp "$INSTALL_DIR/voices/models/en_US-lessac-medium.onnx"* "$INSTALL_DIR/voices/" 2>/dev/null || true

# [4/6] Installing AOC v5 package (pip install -e .)
echo "[4/6] Installing AOC v5 package (pip install -e .)"
python3 -m pip install $PIP_FLAGS -e "$SRC_DIR" || {
  echo "[!] pip install -e . failed — trying with --break-system-packages"
  python3 -m pip install --break-system-packages -e "$SRC_DIR" || true
}

# Copy runtime config + launcher into ~/ai-voice
cp "$SRC_DIR/config.json" "$INSTALL_DIR/config.json" 2>/dev/null || true
cp "$SRC_DIR/install.sh" "$INSTALL_DIR/ai-voice.sh"
chmod +x "$INSTALL_DIR/ai-voice.sh" 2>/dev/null || true
# Also stage C++ sources for optional build
for f in hush_bus_executor.cpp hush_engine_adapters.cpp CMakeLists.txt; do
  [ -f "$SRC_DIR/$f" ] && cp "$SRC_DIR/$f" "$INSTALL_DIR/$f" || true
done
[ -d "$SRC_DIR/systemd" ] && cp -r "$SRC_DIR/systemd" "$INSTALL_DIR/" || true
[ -d "$SRC_DIR/termux-boot" ] && cp -r "$SRC_DIR/termux-boot" "$INSTALL_DIR/" || true

# [5/6] Building HUSH executor (optional)
echo "[5/6] Building HUSH executor (optional)"
if command -v cmake >/dev/null 2>&1; then
  [ -d "$SRC_DIR/processor" ] && cp -r "$SRC_DIR/processor" "$INSTALL_DIR/processor" || true
  [ -d "$SRC_DIR/third_party" ] && cp -r "$SRC_DIR/third_party" "$INSTALL_DIR/third_party" || true
  if [ -d "$INSTALL_DIR/processor" ] && [ -d "$INSTALL_DIR/third_party" ]; then
    ( cd "$INSTALL_DIR" && cmake -B build -S . && cmake --build build -j ) \
      || echo "[!] hush_bus_executor build failed (safe to ignore)"
  else
    echo "[5/6] Skipping executor build (no processor/ or third_party/)"
  fi
else
  echo "[5/6] Skipping executor build (cmake not found)"
fi

# [6/6] Building defense manifest
echo "[6/6] Building defense manifest"
MANIFEST="$INSTALL_DIR/models.manifest"
{
  echo "# AOC v5 models.manifest"
  echo "# generated $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date)"
  echo ""
  find "$INSTALL_DIR/voices" -type f \( -name "*.onnx" -o -name "*.onnx.json" \) 2>/dev/null | while read -r f; do
    sz=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
    echo "voice  $(basename "$f")  $sz"
  done
  echo "aoc_magic_v5  present  1"
  echo "hush_voice  present  1"
  echo "ai-voice.sh  present  1"
} > "$MANIFEST"
NFILES=$(grep -c . "$MANIFEST" || echo 0)
echo "  manifest: $NFILES files"

echo "$BANNER"
echo " Installed to $INSTALL_DIR"
echo ""
echo " Next:"
echo "   aoc-daemon &"
echo "   aoc-voice"
echo "   aoc-magic \"hello\" --out /tmp/h5.wav"
echo "   aoc-magic-engines"
echo "$BANNER"
