#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "SAM 3D requires Linux x86_64 and an NVIDIA CUDA GPU." >&2
  exit 1
fi

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export UV_PROJECT_ENVIRONMENT="$ROOT/data/tools/sam3d-env"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-4}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
test -x "$CUDA_HOME/bin/nvcc"
mkdir -p data/tools
uv sync --locked --inexact --project experiments/sam3d
PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
SOURCE="$ROOT/data/tools/sam-3d-objects"
if [[ ! -d "$SOURCE/.git" ]]; then
  git clone https://github.com/facebookresearch/sam-3d-objects.git "$SOURCE"
  git -C "$SOURCE" checkout f91db411c50efee93d8db7aeb323885650f6f722
fi
"$PYTHON" -c 'from pathlib import Path; from experiments.sam3d.common import verify_source; verify_source(Path("data/tools/sam-3d-objects"))'

# Minimal CUDA images may omit headers already included in PyTorch's CUDA wheels.
CUDA_INCLUDES=$("$PYTHON" -c 'import site; from pathlib import Path; print(":".join(str(p) for root in site.getsitepackages() for p in (Path(root) / "nvidia").glob("*/include")))')
export CPATH="$CUDA_INCLUDES${CPATH:+:$CPATH}"
uv pip install --python "$PYTHON" --no-build-isolation --no-deps \
  flash-attn==2.8.3 \
  'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47' \
  'gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7'
"$PYTHON" "$SOURCE/patching/hydra"

if [[ ! -d data/tools/sam2/.git ]]; then
  git clone https://github.com/facebookresearch/sam2.git data/tools/sam2
  git -C data/tools/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
fi
uv pip freeze --python "$PYTHON" > data/tools/sam3d-environment.txt
export CONDA_PREFIX="$CUDA_HOME"
"$PYTHON" - <<'PY'
import sys
import torch
sys.path[:0] = ["data/tools/sam-3d-objects/notebook", "data/tools/sam-3d-objects"]
from inference import Inference
assert torch.cuda.is_available()
x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all()
print("SAM 3D imports and CUDA kernel passed:", torch.cuda.get_device_name())
PY
