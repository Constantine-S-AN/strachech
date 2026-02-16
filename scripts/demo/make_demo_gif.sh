#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/../.." && pwd)"
tape_path="${script_dir}/vhs.tape"
output_path="${root_dir}/docs/assets/demo.gif"

mkdir -p "${root_dir}/docs/assets"

if ! command -v vhs >/dev/null 2>&1; then
  echo "[demo-gif] vhs is not installed. Skipping GIF generation."
  echo "[demo-gif] Install vhs: https://github.com/charmbracelet/vhs"
  echo "[demo-gif] Expected output path: ${output_path}"
  exit 0
fi

echo "[demo-gif] Recording terminal demo..."
(
  cd "${root_dir}"
  vhs "${tape_path}"
)

echo "[demo-gif] GIF generated: ${output_path}"
