#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "This environment targets Linux x86_64 with an NVIDIA GPU." >&2
  exit 1
fi
command -v uv >/dev/null || { echo "Install uv before running this script." >&2; exit 1; }
nvidia-smi
uv sync --locked --project experiments/mapanything
uv run --locked --project experiments/mapanything python - <<'PY'
import torch
from mapanything.models import MapAnything

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA. Check the container GPU and host driver.")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("BF16 is required by this baseline.")
matrix = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
result = matrix @ matrix.T
torch.cuda.synchronize()
if not torch.isfinite(result).all():
    raise SystemExit("CUDA kernel check produced non-finite output.")
print({"torch": torch.__version__, "cuda": torch.version.cuda,
       "gpu": torch.cuda.get_device_name(0), "model_import": MapAnything.__name__,
       "capability": torch.cuda.get_device_capability(0), "kernel_check": "passed"})
PY
