# Shared frame proposals

This development workflow compares two existing library instances, proposes a
shared direction meaning, and applies that exact meaning to each mesh. It does
not create a category table or automatically admit a new category standard.

See the [two-mug development trial](experiments/2026-10-29-shared-frame.md) for the
first real result: one full-frame proposal and one up-only proposal, with retained
failures and limitations.

The pair matcher receives verified source renders, original photographs and
unverified direction descriptions from the two saved assessments. Display names,
previous identity labels and selected poses are omitted. The direction descriptions
can still reveal identity, so this is not an independent classification benchmark.

The model returns functional identity separately from appearance, matching part
directions, applicability conditions and alternative conventions. A same-function
decision can still omit front or the entire shared frame. The library stores this
as an immutable `MatchProposal`, with a versioned `SharedFrameDefinition` and
checksummed model evidence. Its state remains `proposed`.

## Run a comparison

Import completed evidence-agent results into the [object library](object-library.md)
first. Configure a cloud backend through [environment setup](environment.md).

```bash
uv run --locked --extra canonical --env-file .env.local \
  python -m experiments.instance_matching.worker \
  LEFT_ASSESSMENT_UUID RIGHT_ASSESSMENT_UUID /absolute/path/new-pair-run \
  --backend claude --max-calls 2
```

All referenced evidence is rehashed before dispatch. Up to six source renders and
two original photos per instance are snapshotted. The model cannot execute SQL,
choose artifact paths, overwrite records or establish knowledge. Local validation
requires citations from both actual image sets and existing same-axis direction
IDs. These checks validate references, not semantic correctness.

The output `result.json` records the proposal UUID. `proposal-N.json` preserves the
original reply and request/call hashes. `calls`, `events.jsonl`, `state.json` and
`request.json` retain the execution history. Resume with identical arguments plus
`--resume`; completed results replay without another model call. A dispatch with
no saved response is never automatically retried. Corrections share the original
one-to-three-call budget. There is no provider fallback in this pair workflow.

## Apply the proposed frame

```bash
uv run --locked --extra canonical --env-file .env.local \
  python -m experiments.instance_matching.align \
  OBJECT_UUID PROPOSAL_UUID /absolute/path/views /absolute/path/oa \
  /absolute/path/new-alignment-job \
  --observations /absolute/path/observations \
  --backend claude --max-calls 4 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
```

The source mesh must match one of the proposal's instances. The worker verifies
the stored proposal evidence and both source assessments before using them.
`shared-frame-alignment-v1` reuses the existing evidence agent and its tools. The
shared definition is an explicit task convention: it settles which meaning to
attempt, but cannot settle missing observations or incorrect geometry.

Exported direction hypotheses must retain the exact shared `from_part → to_part`
strings, using `shared-up` and `shared-front` IDs. The harness also retains all
candidate-membership, support-refinement, photo-corroboration and projected-point
checks. It cannot turn a textual agreement into a successful pose. An instance can
remain up-only, unresolved, pending more evidence or fail within its call budget.
The proposal UUID, full definition and source hash are bound into job identity.

The coordinate basis is right-handed: +Y is up; +Z is the supported front projected
perpendicular to up. The two meshes generally need **different rotation matrices**
to express the same part directions in that shared basis.

Generate the existing execution report with:

```bash
uv run --locked --extra canonical \
  python -m experiments.canonicalization.agent_report /absolute/path/alignment-job
```

These alignment jobs remain experiment artifacts. The current library importer
accepts the original `oa-evidence-agent-v3` route, not this new alignment route.
Pending capture continuation is also not wired into this CLI yet; a requested
capture is recorded, not automatically fulfilled or resumed with extra images.

## Pose visualization

User-facing pose previews must show X in red, Y in green and Z in blue. Create
presentation artifacts separately from raw inference inputs:

```bash
uv run --locked --extra canonical python -m experiments.orientation.pose_preview \
  /absolute/path/canonical-render-bundle /absolute/path/new-overlay-directory \
  --scope up_only
```

Use the scope recorded in the actual result. The tool projects axes at the mesh
bounding-box center through the saved camera using the glTF-to-Blender basis and
perspective intrinsics. Unresolved axes are dashed and labeled `?`. A dot/cross
indicates an axis nearly along the viewing ray. Colors identify coordinate axes,
not confidence. Solid axes remain proposals, not independent ground truth.
Candidate-pose render bundles are allowed only with `--scope unresolved`; their
per-view rotations are applied to the overlay axes. The manifest records input,
pose, image and renderer hashes. No raw render is overwritten or fed back with
annotations to the model.

## Compute and evaluation boundaries

Saved colored meshes and matching OA predictions are enough for this workflow:
geometry, storage and Blender rendering run locally; visual decisions use a cloud
API. No rented GPU is required. Fresh SAM 3D reconstruction or fresh OA inference
for a changed input/view set requires its separate model environment.

A shared meaning plus two accepted exports is only a development result. The
projected checks use the same policy's estimated landmarks and do not independently
establish 3D accuracy. Evaluate same-instance random rotations with fresh inference,
then unseen same-function instances, functional lookalikes, missing parts and
symmetries. Rotating cached predictions along with a mesh only tests coordinate
bookkeeping, not model robustness. Pair proposals must not silently become global
standards or leak into held-out annotations.
