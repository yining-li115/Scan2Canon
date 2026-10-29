# Semantic orientation experiment

This experiment connects an existing GLB to the pinned official
[Orient Anything V2](https://github.com/SpatialVision/Orient-Anything-V2) model.
It renders known-camera images, estimates each view independently, maps orientation
predictions back to the original mesh frame, and exports ranked rigid-pose candidates.
The model estimates orientation; it does not generate or repair geometry.

This is a Scan2Canon integration and consensus baseline, not a reproduction of the
complete CanoVerse or CANIS methods. It remains independent of the production worker.
This inference adapter does not perform semantic review. The separate
[open semantic worker](open-canonicalization.md) consumes its results for cloud
visual assessment and bounded evidence requests. Shared category-convention
retrieval remains future work.

The [first GPU run](experiments/2026-10-26-orientation.md) completed on an A100 40GB.
It produced reviewable meshes but conflicting orientation/symmetry predictions;
the semantic decision remains `needs_evidence`.

Use the [controlled diagnostics](orientation-diagnostics.md) to compare the official
entry point, evaluate known rotations, and inspect support-surface upright candidates.

## Local preparation

Install the small CPU geometry extra without GPU dependencies:

```bash
uv sync --extra canonical
uv run --extra canonical python -m experiments.orientation.render \
  data/experiments/runs/sam3d-video-5573-01/mesh.glb \
  data/experiments/inputs/video-5573-orientation-02 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender --views 12
```

Every output directory must be new. On Linux pass the available Blender executable.
The renderer preserves a byte-identical `source.glb`, PNG views, camera rotations,
render settings, file hashes, a contact sheet and a failure-aware manifest. Lighting
uses Cycles CPU with 16 samples and transparent backgrounds. The inference helper
composites alpha onto white. Camera samples cover a sphere without assuming the
input already has an upright orientation. Imported scene lights and cameras are
ignored. Use static GLBs; animation and articulated-state normalization are outside
this experiment.

For the current hole punch, use the existing `video-5573-orientation-02` bundle.
The earlier `-01` render was overexposed and is retained only for diagnostics.

For appearance-aware runs, use `--require-appearance` during rendering and OA
inference. See the [appearance guide](appearance-canonicalization.md) for original
photo admission, appearance checks and complete commands. Existing gray-mesh runs
remain geometry-only controls.

## GPU installation and inference

For the first OA-only inference run, budget one 24 GB NVIDIA GPU, 8 vCPUs,
32 GB host RAM (64 GB preferred), and 100 GB of disk. This is a planning estimate
for our sequential single-view runner, not a measured minimum or an upstream
hardware guarantee. A 40/48 GB GPU provides additional headroom; multiple GPUs
are not required by this runner. Record actual peak memory on the first run.
The existing SAM 3D mesh is reused, so its environment and checkpoints are not
needed on this instance.

Maintain source code and documentation locally. Commit and push tested changes,
then clone or fast-forward pull the same commit on the server. Configure the
isolated GPU environment and download checkpoints on the server. If execution
reveals a code problem, fix it locally and repeat that synchronization process.
Use file transfers only for ignored input/output artifacts. Download and verify
results before releasing an instance.

The OA environment is separate from both the backend and SAM 3D. It targets Linux
x86_64, Python 3.11, PyTorch 2.7.1 and CUDA 12.8 wheels; no driver installation is
performed. Setup verifies the GPU kernel and upstream imports. Download size and
runtime are separate from model inference measurements.

```bash
bash scripts/setup_orientation.sh
experiments/orientation/.venv/bin/python -m experiments.orientation.download
experiments/orientation/.venv/bin/python -m experiments.orientation.infer \
  data/experiments/inputs/video-5573-orientation-02 \
  data/experiments/runs/orientation-video-5573-01
```

Transfer the prepared bundle separately from Git; captures, renders, weights and
results are ignored data. Source code must be committed/pushed before cloning or
pulling on the server. Do not copy credentials or virtual environments.

The pinned checkpoint is `Viglong/OriAnyV2_ckpt` at revision
`34fe2ecbdb987a0fb98437af00600f22a0fdfb99`, file
`demo_ckpts/rotmod_realrotaug_best.pt`, as used by upstream `app.py`.
The README's alternative model repository name was not used. The official source
is pinned to `73b11c9dc83e84daeb563d0c766831f2c66b0a18`. Their license is CC-BY-4.0;
consult upstream terms before redistributing their code or weights.

Inference calls the official preprocessing and decoding helpers and loads the
checkpoint strictly. It stores all 900 logits per view, decoded angles and symmetry
fits, source/input/checkpoint hashes, installed versions, timings and peak GPU memory.
The upstream module imports rembg, but our presegmented renders never invoke its
background model or download its weights. Views are processed sequentially to bound
GPU memory. There is no silent CPU or fake-prediction fallback.

## Coordinate conventions

Column vectors are used throughout. Let:

- `B` map GLB coordinates to Blender: `(x, y, z) -> (x, -z, y)`.
- `C` be the recorded Blender camera-to-world rotation.
- `D` be the camera-to-world rotation from upstream `assets/axis_render.blend`.
  Its camera sits at +X, with local right=+Y, up=+Z, backward=+X.
- `P = Rx(roll) Ry(elevation) Rz(-azimuth)` be upstream's axis-object rotation.

Then `R_oa_from_mesh = P.T @ D @ C.T @ B`. The GLB export permutes OA's
`+X=front, +Z=up` into `+Z=front, +Y=up` with a proper rotation. No reflection or
scale is introduced. Translation centers the input's world-space bounding box.
Physical dimensions remain unverified; this is not an estimate of center of mass.

`calibrate_blender.py` checks the actual official camera, axis meshes, Blender Euler
order and inverse recovery, rather than assuming generic Euler conventions:

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python experiments/orientation/calibrate_blender.py -- \
  data/tools/Orient-Anything-V2 data/experiments/orientation-coordinate-calibration.json
```

This calibration passed locally with Blender 5.1.0 on October 26, 2026. Geometry
tests also verify inverse recovery under input rotations and GLB export without
topology, vertex-color or shape changes. These checks establish coordinate math,
not semantic accuracy of OA predictions.

## Candidate selection and review

```bash
uv run --extra canonical python -m experiments.orientation.solve \
  data/experiments/inputs/video-5573-orientation-02 \
  data/experiments/runs/orientation-video-5573-01 \
  data/experiments/runs/canonical-video-5573-01
uv run --extra canonical python -m experiments.orientation.render \
  data/experiments/runs/canonical-video-5573-01/candidate-01.glb \
  data/experiments/runs/canonical-video-5573-01-preview \
  --blender /Applications/Blender.app/Contents/MacOS/Blender --preset canonical
```

The solver uses the modal symmetry fit and ranks actual view hypotheses by agreement,
retaining up to three distinct hypotheses per mode. When modes tie, it preserves
every tied mode and sets the global `symmetry_fit` and `front_observable` to null.
Candidates record their own symmetry fit and equivalent yaw offsets; they are grouped
by fit and ranked only within that group, not across incompatible conventions.
It does not average conflicting Euler
angles. The default agreement radius is 25 degrees; 60% view support and 75% symmetry
agreement are provisional experiment heuristics, not calibrated confidence thresholds.
They are stored in or described by the report and require evaluation on labeled data.

Fits 1/2/4 compare rotations modulo 360/180/90-degree yaw. Fit 0 compares only the up
axis and always requests more evidence: it may represent unobservable azimuth or
an unreliable fit, and does not establish physical rotational symmetry. Exported
yaw in that case is merely a representative, not a semantically established front.
Symmetry disagreements and low cross-view support also yield `needs_evidence`.
Only modal-fit views vote; all raw predictions remain in the inference report.

`candidate-01.glb` and any alternatives are review artifacts. Their transform matrices,
supporting view IDs, fit counts and equivalence offsets are in `canonicalization.json`.
Even a stable candidate remains `dataset_approval: pending_review`. Its directions
follow OA's learned convention, not a verified category-specific convention such as
"the paper insertion side is the front of a hole punch". Inspect the fixed front,
back, side, top and bottom renders and compare with real evidence before acceptance.

Rendering the same generated mesh from many views adds no real-world observations.
Reconstruction fidelity and held-out real-view evaluation remain separate. A flat
or conflicting prediction must never be converted into automatic dataset approval.

## Validation

```bash
uv sync --locked --extra canonical
uv run --locked --extra canonical pytest
uv run --locked --extra canonical ruff check .
uv run --locked --extra canonical ruff format --check .
uv run --locked --extra canonical mypy src tests experiments
bash -n scripts/setup_orientation.sh
```

Geometry tests require the `canonical` extra; without it they are explicitly skipped.
GPU inference must be checked separately from these CPU tests. The first measured
run is documented above; successful execution does not establish semantic accuracy.
