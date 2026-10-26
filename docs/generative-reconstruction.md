# Generative reconstruction experiment

Use the official [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects)
image-and-mask pipeline as the first generative backbone. The experiment calls its
public `Inference` wrapper with compilation disabled and exports its actual mesh
and Gaussian outputs. It does not replace or integrate with the backend worker yet.
TRELLIS.2 is a possible later comparison, not a prerequisite or a second pipeline stage.

## First verified run

On October 26, 2026, the official SAM 3D pipeline completed on the rented A100 40 GB
using `frame_016.jpg` from the user's video and a SAM 2.1 mask with a manual box.
The local result is under ignored `data/experiments/runs/sam3d-video-5573-01/`.

| Measurement | Result |
| --- | --- |
| Model loading, including first-use auxiliary downloads | 47.0 s |
| Inference | 20.4 s |
| Total runner time, including validation/import/export overhead | 91.6 s |
| Peak allocated CUDA memory | 17.8 GiB |
| Raw mesh | 316,954 vertices; 633,888 triangles |

Installation and the main checkpoint download are outside these timings. This is
one measurement, not a latency guarantee. The exported mesh preserves the broad
punch structure, opening, and separate base. Visual inspection also shows surface
ripples and inferred underside details. No quantitative multi-view accuracy,
metric scale, or semantic canonical orientation has been established. The result
remains pending review. A separate auxiliary cache manifest records the DINOv2
source/weights and MoGe snapshot files loaded during this run.

## Environment

Run on Linux x86_64 with an NVIDIA GPU (upstream specifies at least 32 GB VRAM).
The isolated environment uses Python 3.11, PyTorch 2.5.1 / CUDA 12.1, and the
officially pinned PyTorch3D and gsplat revisions. The backend remains on Python 3.12.
Base Python packages are locked in `experiments/sam3d/uv.lock`; compiled extensions
are installed separately against that PyTorch build. The setup script limits
compilation to four jobs and defaults to A100 compute capability 8.0. Override
`TORCH_CUDA_ARCH_LIST` for other supported hardware.

From the repository root:

```bash
bash scripts/setup_sam3d.sh
```

The script checks out upstream source under ignored `data/tools/`, installs into
`data/tools/sam3d-env`, applies upstream's pinned Hydra compatibility patch, and
records the package inventory. It does not install or change NVIDIA drivers.
An installed CUDA toolkit with `nvcc` is required; set `CUDA_HOME` if necessary.
Use the environment's Python directly after setup. An exact `uv sync` would remove
separately compiled extensions; rerun the setup script to restore them.

## Authorized checkpoint download

The model account must have approved access to
[facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects).
Authenticate in the server terminal using `hf auth login`. A fine-grained token
must allow reading public gated repositories that the account can access.
Do not put tokens in source, command arguments, manifests, or chat messages.

On the current Vast image the interactive terminal uses `HF_HOME=/workspace/.hf_home`.
Non-interactive SSH does not inherit it; use the same `HF_HOME` for downloads and runs.

```bash
export HF_HOME=/workspace/.hf_home
data/tools/sam3d-env/bin/python -m experiments.sam3d.download \
  --checkpoint-dir weights/sam3d
```

The downloader pins the model revision and records SHA-256 hashes. It can resume
an incomplete download and verifies an existing completed bundle before reusing it.
Model code can also request auxiliary weights on its first inference; those requests
use the same authenticated cache. Preserve that cache and record any additional
download revisions before claiming the complete model bundle is reproducible offline.

## Input and mask

Use an upright RGB image from capture normalization. A mask must be the same size,
grayscale, and binary: 255 for the object, 0 for background. Empty, full, mismatched,
or modified masks are rejected. Raw source files are preserved.

For an existing mask, run in the backend environment:

```bash
uv run python -m experiments.sam3d.prepare image.jpg mask.png data/experiments/inputs/object-01 \
  --mask-method "Describe the actual segmentation or annotation method"
```

For the first experiment, SAM 2.1 small generates the mask from a manually annotated
bounding box. The setup script fetches its pinned official source. Its checkpoint
revision, checksum, box, and score are saved with the mask. Inspect `mask-overlay.jpg`
before generation; the score is a model prediction, not a measured segmentation score.

```bash
data/tools/sam3d-env/bin/python -m experiments.sam3d.segment image.jpg \
  data/experiments/inputs/object-segmentation-01 --box 10 20 800 900
```

Replace the example box with pixel coordinates `x_min y_min x_max y_max` for the
actual object. This produces a prepared `input/` subdirectory. The overlay is only
for inspection and is never passed to the reconstruction model. This baseline does
not implement automatic category detection or interactive iOS segmentation.

## Generation

```bash
export HF_HOME=/workspace/.hf_home
data/tools/sam3d-env/bin/python -m experiments.sam3d.run \
  data/experiments/inputs/object-segmentation-01/input \
  data/experiments/runs/sam3d-object-01 --seed 42
```

Every run needs a new output directory. It records failures and preserves completed
outputs rather than overwriting a previous result. Outputs include:

- `mesh.glb`: the official pipeline's mesh with vertex colors; texture baking is disabled.
- `gaussians.ply`: Gaussian splats, not an ordinary scan point cloud.
- `raw-mesh.npz`: decoded mesh vertices and faces before GLB export.
- `predicted-layout.npz`: available predicted rotation, translation, and scale fields.
- `run.json`: versions, input/checkpoint provenance, options, timing, memory, and status.

The native mesh, GLB export, and predicted layout can use different coordinate
conventions. Do not apply layout transforms to the GLB without checking upstream's
conversion. Neither the model's native coordinates nor its predicted pose establish
the category's semantic canonical frame. Scale is not verified against measurements.

## Evaluation boundary

A successful run means that inference and artifact export completed with basic
numeric checks. Dataset approval stays pending. Retain all original video frames;
use other viewpoints to assess silhouette, openings, proportions, and visible parts
after explicitly aligning cameras and object coordinates. Those views are not inputs
to this single-image baseline, and held-out evaluation is not implemented by the runner.

Never label unseen generated surfaces as observations. Store captures, model outputs,
validation evidence, and reviewed canonical orientation separately. Automatic review,
geometric refinement from extra views, semantic canonicalization, and publication are
future adapters, not implicit consequences of generating a plausible mesh.
