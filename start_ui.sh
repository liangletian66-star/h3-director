#!/usr/bin/env bash
# h3-director 导演台（Linux / macOS）
set -e
cd "$(dirname "$0")"

if [ ! -f config.json ]; then
  cp config.example.json config.json
  echo "[!] 已生成 config.json，请把 comfy_dir 改成你的 ComfyUI 根目录（能看到 input/ output/ 的那一层）"
  exit 1
fi

PORT="${1:-8190}"
echo "h3-director 导演台: http://127.0.0.1:${PORT}"
python3 ui/h3d_server.py --port "$PORT"
