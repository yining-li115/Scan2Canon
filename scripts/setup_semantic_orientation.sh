#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "Semantic model inference requires Linux x86_64 with CUDA." >&2
  exit 1
fi
uv sync --locked --project experiments/semantic_orientation
python_bin=experiments/semantic_orientation/.venv/bin/python
source_dir=data/tools/SoFar
revision="$($python_bin -c 'from experiments.semantic_orientation.assets import config; print(config()["source_revision"])')"
mkdir -p data/tools
if [[ ! -d "$source_dir" ]]; then
  git clone https://github.com/qizekun/SoFar.git "$source_dir"
  git -C "$source_dir" checkout --detach "$revision"
fi
"$python_bin" - <<'PY'
from pathlib import Path
import torch
from experiments.semantic_orientation.assets import verify_source
verify_source(Path('data/tools/SoFar'))
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable')
print({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)})
PY
uv pip freeze --python "$python_bin" > data/tools/semantic-orientation-environment.txt
