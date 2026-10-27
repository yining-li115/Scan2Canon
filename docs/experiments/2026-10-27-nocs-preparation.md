# NOCS pilot: source audit and rotation inputs

This local preparation covers the 12 NOCS instances in
[Mini v1](../datasets/mini-v1.md). It creates shape-only inputs for orientation
inference and separate reference annotations. It does not run OA, PointSO, VLM,
or reconstruction models, and it does not approve assets for dataset publication.

## Coordinate audit

Each source mesh was rendered from all six signed coordinate axes and one oblique
camera: **84 audit images**. Labels such as `Source +X` identify the viewer's
position in the original mesh coordinates, not a predicted semantic front.
The original OBJ geometry and coordinates were preserved in the audit GLBs.

The pinned NOCS `compute_RT_degree_cm_symmetry` implementation compares transformed
+Y vectors for bottles, bowls, and cans. Its original image benchmark also ignores
mug yaw when `handle_visibility == 0`. Our complete-mesh source views show the two
mugs' handles; this experiment is not a reproduction of the occluded-image task.

Source-code inspection and visual inspection of the rendered meshes yielded:

| Category | Source up | Reference direction | Scoring scope |
| --- | --- | --- | --- |
| Bottle, bowl, can | +Y | Yaw ignored under the upstream category metric | Up only |
| Camera | +Y | Lens facing +X | Full frame |
| Laptop | +Y | Horizontal keyboard edge toward user, +X | Full frame |
| Mug | +Y | Handleward +X, an explicit reference convention | Full frame |

The project uses +Y up and +Z reference front. The fixed source-to-project rotation
is therefore:

```text
C = [ 0  0 -1 ]
    [ 0  1  0 ]
    [ 1  0  0 ]
```

This maps source +X to project +Z, preserves up, and has determinant +1. Without
this conversion, comparing source axes directly to project axes would introduce
a 90-degree convention error for directional categories. The mug convention is
handle-based, not a claim that all cups have a universal natural front. Laptop
front is horizontal; it is not the tilted screen normal.

The versioned [conventions](../../experiments/datasets/nocs-conventions-v1.json)
bind these rules to exact source mesh hashes and audit evidence. They are checked
dataset-coordinate references, **not independent human annotations or measured
physical poses**. No tested orientation-model predictions were used to assign them.

## Inputs and answer separation

Each of the 12 objects has five deterministic random SO(3) rotations, sampled from
normalized Gaussian quaternions with per-object PCG64 seeds derived from seed 17.
This produces **60 inputs**, with 30 development and 30 holdout cases. They remain
12 independent objects. Reordering the catalog does not change their rotations.

For source vertices `p`, bounding-box center `c`, and sampled rotation `Q`:

```text
input_vertex = Q @ (p - c)
input_to_project = C @ Q.T
```

The rotation is baked into vertices; GLB scene-node transforms are identity.
Face indices are preserved. Every exported GLB was loaded again and inverted;
vertex recovery is checked with absolute tolerance 1e-6 in source mesh units.
The input index contains no applied rotation, source center, reference frame, or
seed. Source meshes and answer-bearing manifests are kept outside its directory.

Each input is rendered with the existing twelve-view sphere preset at 518 pixels,
using Cycles CPU with 16 samples and known camera transforms: **720 inference
images**. Camera extrinsics describe the input coordinate frame; they are necessary
geometry evidence and do not encode the source-to-input rotation.

For the six axial-metric objects, scoring records contain only the expected up
vector; their full-frame rotation and front are null. The other six have a full
reference rotation. All 60 cases have an up reference. This prevents an arbitrary
yaw representative from becoming an accidental full-frame ground truth.

## Appearance and geometry limitations

The selected OBJ files reference missing material files. Inputs use explicitly
uniform gray vertex colors rather than invented or default texture evidence.
Missing bottom surfaces and other scan defects remain visible; no holes were
filled and no surfaces were regenerated. The meshes are non-watertight when loaded
without welding/processing. A later support-plane baseline must label any
convex-hull or uniform-density center-of-mass approximation as an assumption.

This is a controlled shape-only evaluation. Do not interpret differences from
textured real captures as orientation-model differences alone. Upstream category
symmetry is an evaluation convention, not a claim that every reconstructed vertex
or appearance is exactly symmetric.

## Reproduction

Use the locked CPU environment with the `canonical` extra and an installed Blender.
Run from the repository root. Preparation and audit refuse existing destinations.

```bash
uv run --locked --extra canonical python -m experiments.datasets.nocs audit \
  experiments/datasets/mini-v1.json \
  data/datasets/scan2canon-mini-v1/nocs-audit-02 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender

uv run --locked --extra canonical python -m experiments.datasets.nocs prepare \
  experiments/datasets/mini-v1.json \
  data/datasets/scan2canon-mini-v1/nocs-rotations-02 --count 5 --seed 17

uv run --locked --extra canonical python -m experiments.datasets.references \
  data/datasets/scan2canon-mini-v1/nocs-rotations-02 \
  experiments/datasets/nocs-conventions-v1.json \
  data/datasets/scan2canon-mini-v1/nocs-rotations-02/scoring-references.json

uv run --locked --extra canonical python -m experiments.datasets.nocs render \
  data/datasets/scan2canon-mini-v1/nocs-rotations-02 \
  data/datasets/scan2canon-mini-v1/nocs-renders-01 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender --views 12
```

`nocs-rotations-02/inputs/` and `nocs-renders-01/` are inference inputs.
`references.json`, `scoring-references.json`, and `nocs-conventions-v1.json` belong
only to preparation/evaluation. Do not give them to an inference agent or model.
The conventions pin this audit's evidence hashes; a new audit is a new record and
does not silently replace those references.

Rendering can reuse completed cases only after checking input, image, renderer,
and settings consistency. A partial or corrupted render bundle is rejected and
retained; archive that failed bundle before retrying, or use a fresh destination.
No completed-stage outputs are silently overwritten.

The first audit attempt (`nocs-audit-01`) failed when Blender crashed under the
filesystem sandbox; its failed manifest is retained. The successful audit is
`nocs-audit-02`, with a snapshot of its renderer. The initial preparation
`nocs-rotations-01` is retained; `nocs-rotations-02` records the finalized runner.

## Next experiment

Run OA V2 on the development cases first. Compare the raw orientation baseline to
the support-assisted variant, then freeze conventions and decision thresholds
before examining holdout predictions. Report results by object/category and metric
scope, with equivariance separate from semantic accuracy. There are no model
accuracy or automatic-acceptance results from this preparation stage.
