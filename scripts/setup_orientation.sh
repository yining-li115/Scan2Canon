#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "OA inference requires Linux x86_64 and an NVIDIA CUDA GPU." >&2
  exit 1
fi
command -v uv >/dev/null || { echo "Install uv first." >&2; exit 1; }
nvidia-smi
uv sync --locked --project experiments/orientation
python_bin=experiments/orientation/.venv/bin/python
source_dir=data/tools/Orient-Anything-V2
revision="$($python_bin -c 'from experiments.orientation.common import config; print(config()["source_revision"])')"
mkdir -p data/tools
if [[ ! -d "$source_dir" ]]; then
  git clone https://github.com/SpatialVision/Orient-Anything-V2.git "$source_dir"
  git -C "$source_dir" checkout --detach "$revision"
fi
"$python_bin" - <<'PY'
import sys
from pathlib import Path
import torch
from experiments.orientation.common import verify_source

source = Path('data/tools/Orient-Anything-V2').resolve()
verify_source(source)
sys.path.insert(0, str(source))
from vision_tower import VGGT_OriAny_Ref
from utils.app_utils import preprocess_images

if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable.')
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
x = torch.randn(256, 256, device='cuda', dtype=dtype)
if not torch.isfinite(x @ x.T).all():
    raise SystemExit('CUDA matrix test failed.')
print({'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
       'cuda': torch.version.cuda, 'imports': 'passed', 'kernel': 'passed'})
PY
uv pip freeze --python "$python_bin" > data/tools/orientation-environment.txt
