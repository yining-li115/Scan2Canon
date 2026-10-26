# Server experiments

The first baseline uses the [official MapAnything implementation](https://github.com/facebookresearch/map-anything), with its image loading and `model.infer` API. It does not feed COLMAP estimates into the model. This keeps the first experiment image-only and leaves the local COLMAP run as a separate reference.

## Scope and verification

The runner exports per-view depth, intrinsics, camera-to-world poses, confidence, valid masks, RGB, and world points. A bounded point-cloud preview is provided as PLY and GLB. GLB is a file format here, not evidence of a fused or watertight mesh. Background points remain; semantic canonicalization, object segmentation, and scale verification are not performed.

Local tests cover sample preparation, checksums, portability, failure records, and refusing to overwrite outputs. GPU installation, checkpoint loading, BF16 CUDA execution, and eight-view inference/export were verified on an A100 40 GB on October 26, 2026. Reconstruction quality still requires evaluation. A run marked `succeeded` means inference and exports completed, not that geometric accuracy or canonical orientation passed an evaluation.

### First verified run

Code revision: `52185d84dc584920d8dd92d6dfd7d3f0da8a4608`. Local run directory: `data/experiments/runs/mapanything-video-5573-8-01/` (excluded from Git). The inputs are eight uniformly sampled frames from a real phone video, normalized to JPEG; upstream preprocessing produced 294 × 518 pixels per view.

| Measurement | Result |
| --- | --- |
| GPU / environment | A100 40 GB, compute capability 8.0, Python 3.12.3, PyTorch 2.7.1+cu128 |
| Model loading | 24.18 s |
| Synchronized model inference | 1.44 s |
| Entire runner, including validation, loading, and export | 35.54 s |
| Peak PyTorch allocated / reserved memory | 7.42 / 7.58 GiB |
| Valid point samples across eight views | 1,170,791 (not deduplicated) |
| Preview point count | 200,000 |

All eight prediction files were readable. Valid depths were positive and finite; valid world points were finite; camera rotation matrices were orthonormal with determinant approximately +1. These are numeric integrity checks only. The scene includes the desk and background, and the object is not isolated or converted into a complete mesh. This single first-run measurement is not a steady-state performance benchmark. The predictions and logs were copied back to the development Mac.

Comparisons remain independent: VGGT on the same frames is planned;
[SAM 3D Objects](generative-reconstruction.md) has a verified single-image GPU run;
[Orient Anything V2](canonicalization.md) has an isolated orientation adapter with
local coordinate/export verification and GPU inference still to be verified.

## Pinned environment

- Linux x86_64, Python 3.12, NVIDIA GPU. Verified on A100 40 GB; RTX PRO 5000 Blackwell remains an untested target for this pinned environment.
- PyTorch 2.7.1, torchvision 0.22.1, torchaudio 2.7.1, using the official CUDA 12.8 wheels. Torchaudio is a transitive UniCeption dependency.
- MapAnything source: `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9`.
- Research checkpoint: `facebook/map-anything`, revision `a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa`, **CC-BY-NC-4.0**. This is a research baseline; do not assume this checkpoint is suitable for commercial app distribution. The Apache checkpoint is a different experiment and is not substituted silently.
- DINOv2 Torch Hub source: `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.

Python dependencies are resolved in `experiments/mapanything/uv.lock`. The NVIDIA driver, OS image, and CUDA compiler are outside that lock. A host driver compatible with CUDA 12.8 is required; `nvidia-smi`'s CUDA field describes driver capability, not the installed Python runtime. The setup script exercises an actual BF16 CUDA matrix multiplication before checkpoint downloads.

Use persistent storage for the checkout, environment, weights, and results. Around 200 GB disk gives room for dependencies, caches, captures, and later baselines. Start at 8 views; larger frame counts still require memory measurements. On Vast.ai, `/workspace` is not necessarily a persistent volume: container storage can disappear when the instance is destroyed or recycled, so copy results off the instance first.

## 1. Prepare the input on the Mac

Run from the repository root in the normal backend environment:

```bash
uv sync --locked
uv run python -m experiments.mapanything.prepare \
  data/captures/video-5573/frames-dense \
  data/experiments/inputs/video-5573-8 --views 8
```

The command uniformly samples lexicographically ordered filenames, includes the endpoints for multiple views, and normalizes selected images using the same HEIC/JPEG/PNG decoder as capture import. Use zero-padded filenames in capture order. It records original and normalized hashes and refuses duplicate normalized frames or an existing destination. The original directory is untouched.

`--views 1` is supported for a single-image baseline; the backend's separate minimum-three-image capture rule does not apply here. Change the output name when preparing another subset. This is coverage by index, not a blur or viewpoint-quality selector. The prepared manifest is portable and contains no absolute source paths.

Prepared captures and model outputs are ignored by Git. Cloning the code does not transfer the input images. After obtaining the server's SSH host, port, and checkout path, transfer only the prepared folder, for example:

```bash
rsync -av -e 'ssh -p PORT' \
  data/experiments/inputs/video-5573-8 \
  USER@HOST:/workspace/Scan2Canon/data/experiments/inputs/
```

Replace `PORT`, `USER`, `HOST`, and `/workspace` with the actual instance details. Create the destination parent directory on the server first. The prepared JPEGs omit EXIF/XMP; do not upload the raw originals unless they are needed for a later experiment.

## 2. Clone and install on the server

Requires `git`, `uv`, and NVIDIA GPU access inside the server/container. Run these after the local changes have been pushed to GitHub:

```bash
cd /workspace
git clone https://github.com/yining-li115/Scan2Canon.git
cd Scan2Canon
git rev-parse HEAD
mkdir -p data/experiments/inputs
bash scripts/setup_mapanything.sh
```

If the repository is already cloned, update it before setup. Run all following commands from the checkout root. `uv` creates `experiments/mapanything/.venv`; no activation or changes to the backend environment are needed. Do not run the GPU setup on the Mac. A missing driver, unsupported GPU kernel, or package import failure stops setup and must be fixed before continuing.

## 3. Download the checkpoint separately

```bash
uv run --locked --project experiments/mapanything \
  python -m experiments.mapanything.download \
  --checkpoint-dir weights/mapanything
```

This downloads the pinned model configuration, weights, and model card. It also prepares the pinned DINOv2 source at the cache name requested by upstream UniCeption. The pinned MapAnything loader disables redundant DINOv2 pretrained-weight downloads. Interrupted model downloads can be retried with the same command. A mismatched existing DINOv2 checkout is rejected rather than reset.

The bundle manifest records revisions, license, and SHA-256 checksums. Inference verifies them and loads weights locally. The upstream Torch Hub implementation may still attempt a GitHub branch lookup; the checked local cache provides its fallback. This is not an air-gapped deployment promise.

## 4. Run the first smoke test

```bash
uv run --locked --project experiments/mapanything \
  python -m experiments.mapanything.run \
  data/experiments/inputs/video-5573-8 \
  data/experiments/runs/mapanything-video-5573-8-01 \
  --checkpoint-dir weights/mapanything
```

The runner uses the official fixed-mapping 518-resolution preprocessing, DINOv2 normalization, BF16, and memory-efficient inference with dense-head minibatch size 1. This minibatch option does not divide multiview attention into independent single-frame jobs. It requires CUDA and will not silently fall back to CPU or MPS.

Each output directory contains:

| File | Contents |
| --- | --- |
| `run.json` | Status, errors, settings, dependency versions, lock/runner hashes, timing, GPU and peak PyTorch memory |
| `input-manifest.json` | Exact input order and file hashes |
| `checkpoint-manifest.json` | Checkpoint identity and file hashes |
| `nvidia-smi.txt` | Driver and device snapshot |
| `view-0000.npz`, ... | Full-resolution per-view prediction arrays at model resolution |
| `points.ply`, `preview.glb` | Up to 200,000 sampled valid points in the model's world frame |

NPZ keys: `depth_z`, `intrinsics`, `camera_to_world`, `world_points`, `image_rgb`, `confidence`, and `valid_mask`. Intrinsics refer to resized/cropped model inputs. `world_points` are computed using the official depth-to-world helper, matching the official demo; they are not semantic canonical coordinates. `valid_mask` combines upstream geometric filtering and finite-value checks, not an object mask. Use the saved RGB and intrinsics together; original-image pixel coordinates differ.

Model loading and synchronized inference are timed separately. First inference can include compilation/warmup; do not present it as steady-state throughput. Peak allocated/reserved memory describes PyTorch's allocator, not total system GPU memory. Model metric scale remains an unverified prediction.

Errors return a nonzero exit status and preserve `run.json` and any partial outputs. Use a new output directory for retries. A hard kill or power loss can leave status `running`; only `succeeded` confirms completed exports. The experiment does not resume an interrupted forward pass.

## 5. Inspect before increasing scope

Check that all selected frames produced outputs, inspect the point cloud and camera consistency, and compare visible object structure with the input images. Record failures as well as successful cases. Increase to 16 views in a separately named subset and output directory after checking memory. Hold out other video frames for later geometry evaluation; the smoke test alone is not a benchmark.

Only after GPU reproduction and quality checks should this runner become a production reconstruction adapter. Backend jobs intentionally remain blocked at `reconstruct` until that integration is implemented.
