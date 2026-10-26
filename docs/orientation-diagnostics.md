# Orientation diagnostics

These experiments diagnose the first OA V2 run before changing its model or
accepting a canonical orientation. They do not change the production worker.

The [executed control and support-plane results](experiments/2026-10-26-orientation-diagnostics.md)
record official-entry-point parity, known-pose errors, and the remaining front ambiguity.

## Known-pose and integration controls

`experiments.orientation.controls` authors a simple chair with +Y up and +Z front,
then renders it both unchanged and after a recorded rotation. It also renders
the actual generated hole-punch mesh before and after the same rotation.
The procedural chair is explicitly synthetic test geometry, not reconstructed
user data or a standard benchmark. A single fixture cannot establish general accuracy.

```bash
uv run --locked --extra canonical python -m experiments.orientation.controls prepare \
  data/experiments/runs/sam3d-video-5573-01/mesh.glb \
  data/experiments/inputs/orientation-controls-01 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
```

After committing local changes, pull the same commit on the GPU server and transfer
the ignored input directory. For each of `chair-base`, `chair-rotated`, `punch-base`,
and `punch-rotated`, run:

```bash
experiments/orientation/.venv/bin/python -m experiments.orientation.infer \
  data/experiments/inputs/orientation-controls-01/chair-base \
  data/experiments/runs/orientation-controls-01/chair-base --verify-official
```

`--verify-official` compares our forward-pass logits and decoded values with the
pinned upstream `inf_single_case` entry point on the same RGBA render. It does not
invoke rembg or compare background-removal behavior. A mismatch fails the run.
Timings include the extra parity pass; do not compare them to ordinary inference.

Download predictions before evaluating locally:

```bash
uv run --locked --extra canonical python -m experiments.orientation.controls evaluate \
  data/experiments/inputs/orientation-controls-01 \
  data/experiments/runs/orientation-controls-01 \
  data/experiments/runs/orientation-controls-01/evaluation.json
```

The chair has an authored target frame, so it supports absolute rotation and up-axis
errors. The hole punch has no accepted canonical frame, so only equivariance is
measured. For a known input rotation Q, ideal predictions satisfy R_rotated Q = R_base.
The comparison pairs the same sampled camera index; the visible surface changes
when the object rotates. Full-frame errors do not discount model-predicted symmetry.
Consistency alone does not establish a correct semantic frame.

## Support-surface candidates

[CanoVerse, section 3.1](https://arxiv.org/html/2603.07144v1#S3.SS1) separates
vertical and horizontal alignment and uses support-surface candidates with semantic
selection. Our diagnostic implements a smaller experiment using
[trimesh stable poses](https://trimesh.org/trimesh.poses.html), followed by OA up-axis
agreement. It is not a CanoVerse reproduction and has no VLM selection stage.

```bash
uv run --locked --extra canonical python -m experiments.orientation.support \
  data/experiments/inputs/video-5573-orientation-02 \
  data/experiments/runs/orientation-video-5573-01 \
  data/experiments/runs/support-video-5573-01
```

The computation uses the full mesh's convex hull and its uniform-density center
of mass. Neither density nor physical mass distribution has been measured, and
generated underside geometry is unverified. Quasi-static probabilities are geometric
proxies, never semantic confidence. No maximum-area-plane assumption is used.

All predicted up directions vote independently of the predicted yaw symmetry.
Candidates are sorted by support within 25 degrees, median up disagreement, then
the quasi-static proxy. These thresholds are provisional. Similar up axes within
8 degrees are deduplicated for previews; every raw support plane remains in the report.
Opposite up directions are distinct candidates. Up to six GLBs are exported.

Each GLB preserves shape and scale. Its lowest vertex touches Y=0; XZ is centered.
Yaw comes from projecting one nearby OA front prediction onto the support plane,
so it is an arbitrary review representative until front semantics are verified.
Every result remains `needs_review`, `pending_review`, and `front_established: false`.
Render the exports using the existing `experiments.orientation.render` command
with `--preset canonical`, then compare with real frames.

## Local checks

The root `canonical` extra includes SciPy and NetworkX for trimesh stable poses.
GPU dependencies remain isolated in `experiments/orientation`.
Run the repository's full Ruff, mypy and pytest checks with `--extra canonical`.
Tests cover known input rotations, retained upside-down stable poses, contact with
Y=0, and the distinction between accuracy and equivariance metrics.
