# OA V2 with open semantic evidence

This standalone route replaces the six-category input requirement with an instance
identity hypothesis. OA V2 supplies learned orientation candidates; a cloud visual
model separates function, appearance and visible parts, then assesses rendered
candidates. The existing geometry/rules worker remains a separate baseline.
For model-selected evidence actions and user-capture continuation, use the
[evidence agent](canonicalization-evidence-agent.md). This document describes the
fixed identity/pose/opposite-view workflow for comparison.

```text
Existing GLB, including its available colors/materials
    → known-camera renders → pinned OA V2 inference
    → verified OA predictions in the original mesh coordinate frame

Source renders + optional original photographs
    → VLM: functional identity, appearance, parts, proposed up/front meaning
    → if identity is uncertain: save hypotheses and request specific evidence
    → otherwise: render OA-derived pose candidates
    → VLM: select a candidate, accept up only, or abstain
    → if abstaining: one additional round with opposite views
    → validate decision → reversible canonical GLB proposal, or unresolved report
```

A tomato-shaped object can have `functional_identity: kitchen timer` and
`appearance_description: tomato-shaped enclosure`. The model must support the
function with visible evidence; decorative shape alone cannot establish a timer.
This representation is an instance proposal, not a validated category taxonomy.
It does not yet guarantee that different objects receive consistent class names
or the same semantic front definition. Shared, versioned convention retrieval
and category-consistency evaluation remain subsequent work.

## Prepare and run

The source mesh must be a static GLB. Use the root locked CPU environment and
Blender to prepare 3–12 views (12 recommended):

```bash
uv sync --locked --extra canonical
uv run --locked --extra canonical python -m experiments.orientation.render \
  /absolute/path/object.glb data/experiments/inputs/object-open-views \
  --views 12 --blender /Applications/Blender.app/Contents/MacOS/Blender
```

Run the existing pinned OA inference adapter in its separate NVIDIA GPU
environment. Follow [OA setup](canonicalization.md#gpu-installation-and-inference)
for server installation and checkpoint download. Transfer the complete view bundle
without modifying its manifest, then run on that server:

```bash
experiments/orientation/.venv/bin/python -m experiments.orientation.infer \
  data/experiments/inputs/object-open-views \
  data/experiments/runs/object-open-oa
```

Copy the completed inference directory back alongside the original view bundle.
The next command runs locally with Blender and the Gemini configuration described
in [environment setup](environment.md). It needs no local CUDA installation and
makes at most three model requests:

```bash
uv run --locked --extra canonical --env-file .env.local \
  python -m experiments.canonicalization.open_worker \
  data/experiments/inputs/object-open-views \
  data/experiments/runs/object-open-oa \
  data/experiments/runs/object-open-canonical \
  --backend gemini --max-calls 3 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
```

Use `--blender blender` on Linux. `--backend responses --model MODEL_ID` is an
explicit alternative. The local Qwen selection adapter does not support this
route's semantic schemas. Missing OA predictions cause failure; there is no silent
PCA substitute. The CLI consumes saved OA results, rather than provisioning a GPU
or launching inference remotely.

For the saved hole-punch experiment, the matching inputs are
`data/experiments/inputs/video-5573-orientation-02` and
`data/experiments/runs/orientation-video-5573-01`. These ignored data directories
must be transferred separately from Git.

## Appearance and evidence

The renderer imports the GLB's available appearance. The exporter applies a rigid
scene transform and preserves colors, UVs and materials; it does not generate a
new texture. A colorless mesh stays colorless. SAM 3D's generated appearance is
still a prediction, including surfaces not observed in the source photo.

Add `--observations /absolute/path/bundle` for up to eight verified original
photographs. Use the existing [photo manifest format](canonicalization-agent.md#supplemental-photographs).
The bundle must be bound to this exact mesh hash. Its legacy `category` metadata
is used only to validate/copy the bundle, never as a model identity label or a
lookup into the baseline's six categories. Original photos are labeled `photo:ID`;
mesh renders are `source:ID`, `candidate:ID` and `opposite:ID`.

Photos remain unregistered: the worker does not estimate their camera poses.
A selection citing photos must also describe visible mesh correspondence and cite
the selected mesh candidate plus a contrasting view. This is a checkable evidence
claim, not an independently verified registration or guarantee against hallucination.
Source photos do not directly produce executable mesh-space rotations.

## Direction and decision boundaries

- Each OA prediction is converted using its recorded render camera. Candidate
  rotations map the original GLB into the project convention: **+Y up, +Z front**.
- Symmetry fits 2 and 4 add their yaw alternatives. Fit 0 keeps an up proposal but
  cannot authorize a full frame. These fits are model predictions, not verified
  physical symmetry labels.
- The worker prioritizes diverse rotations and retains minority symmetry modes,
  with at most 12 candidates. Available and omitted hypothesis counts are recorded.
  Candidates can duplicate one another when distinct source views agree. There
  is no claim that this finite pool contains the correct semantic pose.
- The VLM can choose an existing candidate only. It cannot supply a transformation
  matrix. Unknown IDs, invented image labels and invalid decisions fail validation.
- `full_frame` requires a resolved instance hypothesis, a unique proposed front,
  and a candidate with an OA yaw prediction. `up_only` exports one representative
  pose while explicitly leaving yaw unspecified. `unresolved` exports no mesh.
- The geometry exporter checks proper rotation, topology counts and inverse vertex
  recovery. These are mechanical checks; they do not establish semantic correctness.
- Every result has `dataset_approved: false`. Instance hypotheses never become
  registered category conventions automatically.

## Recovery and artifacts

The default budget is one identity call plus up to two pose calls. `--max-calls 2`
disables the extra pose round. Uncertain identity stops after the first call.
The second pose call adds opposite renders; it does not recapture hidden surfaces
or repair geometry. At most 32 images are included in any request.

Each request is journaled before dispatch. Responses, source snapshots, candidate
renders and exported artifacts have hashes. Repeat the same command with
`--resume` to reuse completed calls and stages. Inputs, model identity, code
versions and call budget must match. Tool stages allow at most two attempts.
A network failure without a saved response remains counted and is not automatically
retried, since the provider may already have processed it. An invalid saved model
response is preserved and fails again rather than being silently overwritten.

The command prints the path to `work/export-N/canonicalization.json`. It records
instance semantics, choice, OA candidate provenance, unresolved axes, call count,
actual policy identity and, when exported, the GLB hash and inverse transform.
`status: succeeded` means execution completed, not that a semantic label passed
independent evaluation. This worker is not yet connected to the capture API/iOS app.

## Validation status

Automated tests use clearly labeled scripted model/render fixtures to exercise
coordinate conversion, symmetry restrictions, original-photo evidence, decision
validation, bounded abstention, interrupted export recovery and response integrity.
Actual GLB exports verify reversible geometry plus preserved vertex colors and
embedded texture/UV data. Both cloud transports are tested with the new schemas.
These tests measure implementation behavior, not model accuracy.

The saved real OA predictions can be reused for integration checks without renting
a GPU. Compare this route with the geometry baseline on identical held-out meshes
and rotations before claiming an improvement in accuracy or manual-review rate.

The [implementation validation record](experiments/2026-10-27-open-oa-integration.md)
documents the real Blender integration, the first live evidence-label failure,
and the corrected run's transport timeout. Identity inference executed, but the
complete corrected cloud route still needs a successful live validation.
