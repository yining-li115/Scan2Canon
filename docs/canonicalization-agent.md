# Mesh canonicalization agent

This experiment turns an existing GLB into a rigidly transformed canonical
proposal. It does not reconstruct missing surfaces or certify dataset admission.
The optional GPU models and the interactive visual agent are separate policies.
The interactive agent can execute the repository skill at
`.agents/skills/mesh-canonicalization/SKILL.md`; the iOS/backend does not yet run
that agent automatically.

For automatic decision generation and resumable execution, use the
[standalone worker](canonicalization-worker.md). It shares the validated tools
below and records `automated_model_policy` plus the actual backend/model in its
run provenance. Interactive, cloud and local-model results remain separate policies.

## Flow

1. Preserve the source mesh and render a verified input bundle.
2. Optionally collect OA V2 proposals. View consensus is diagnostic, not proof.
3. Generate upright alternatives from surface PCA and convex-hull support planes.
4. Select an upright alternative from visual part evidence, or describe the
   missing evidence. Keep category conventions separate from instance geometry.
5. For a category with a reference front, generate four horizontal PCA direction
   candidates and select the view facing that direction. If yaw is unspecified,
   skip this stage. An unresolved front does not invalidate a useful up-only output.
6. Validate the decision and export. Check SO(3), unchanged vertex/face counts,
   inverse-transform vertex error, source hashes and stage bindings. Keep the
   selected rotation, translation, inverse, evidence and tool versions.

`canonical.glb` uses +Y up and +Z front where established, with unchanged scale.
Its horizontal origin is the input bounding-box center; its lowest vertex rests
at Y=0. `scope` distinguishes `full_frame`, `up_only`,
`up_only_front_pending`, and `unresolved`. An unspecified yaw is a representative,
not an asserted semantic direction. Unknown physical scale stays unknown.

## Tools and contracts

Use the root locked environment with the `canonical` extra. Blender is needed for
rendering, but GPU libraries are not needed for geometric tools or exports.

```bash
uv run --locked --extra canonical python -m experiments.canonicalization.workflow prepare \
  object.glb mug run/up --blender blender
```

This direct GLB path does not require OA inference or a separate input render pass.
The Python preparation tool also accepts an existing verified render bundle.

Inspect `run/up/poses/contact-sheet.jpg` and, when needed, its original PNGs.
Use `experiments/canonicalization/conventions.json` as the semantic definition.
Save a decision JSON conforming to `workflow.Decision`:

```json
{
  "schema_version": 1,
  "actor": "interactive_visual_agent",
  "case_sha256": "<SHA-256 of upright.json or front.json>",
  "rules_sha256": "<SHA-256 of conventions.json>",
  "model_calls": 1,
  "selection": {
    "decision": "selected",
    "candidate_id": 0,
    "evidence_view_ids": [0, 1],
    "visible_evidence": "Describe the actual parts establishing this candidate and contrasting another.",
    "missing_evidence": ""
  }
}
```

The ID above illustrates the schema; it is not a recommended candidate. A
`needs_evidence` decision must have a null candidate and an actionable missing
evidence description. Unknown IDs, arbitrary matrices, missing cited evidence,
changed inputs and inconsistent up/front choices are rejected.

### Supplemental photographs

An unresolved case can use identity-matched photographs as additional part
evidence. Keep its previous decision intact and start a new diagnostic run.
Unregistered photographs identify parts but do not supply a mesh-space rotation:
the agent must explain how those parts correspond to the rendered mesh candidates.
If that correspondence is missing, keep the case unresolved.

`experiments.datasets.nocs_photos.prepare` creates a local evidence bundle from
explicit NOCS RGB frame stems, instance masks and identity metadata. It checks the
instance/category match and visible mask pixels, preserves the source files and
creates crops with recorded bounds. It does not open coordinate maps or pose labels.
Only use an instance association established by the dataset catalog; the helper's
hash binding is not an automatic identity-recognition model.

Store the bundle inside the candidate directory and include this optional field
in the decision:

```json
{
  "supporting_evidence": {
    "manifest": "observations/manifest.json",
    "sha256": "<manifest SHA-256>",
    "image_ids": ["scene_2-0000"],
    "mesh_correspondence": "Explain the visible part correspondence, or why it remains missing."
  }
}
```

The controller verifies the manifest, mesh/category binding, cited image IDs and
every recorded source/crop hash before accepting the decision. The selected
rotation must still come from the geometric candidates. Count the earlier
selection round in `model_calls`; additional evidence does not reset its budget.
See the [photo-evidence diagnostic](experiments/2026-10-27-photo-evidence.md).

```bash
uv run --locked --extra canonical python -m experiments.canonicalization.workflow front \
  run/up run/up-decision.json run/front --blender blender

uv run --locked --extra canonical python -m experiments.canonicalization.workflow finish \
  run/up run/up-decision.json run/result \
  --front-case run/front --front-decision run/front-decision.json
```

Omit both front arguments for an up-only result or an upright abstention. New
outputs require a fresh directory. Failures remain recorded rather than silently
replacing a completed stage.

For the standalone local Qwen policy, run `experiments.canonicalization.semantic`
in the isolated semantic environment. It verifies weights, uses normal and
reversed candidate order, and enforces two calls per case (maximum 60 per batch).
It produces semantic proposals, not automatically approved decisions. Current
development results show that this 7B model can fail to recognize visible parts.
Do not present the interactive visual agent's results as that model's results.

The interactive workflow allows at most two declared selection rounds per stage,
four total. The tool controller validates those declarations and evidence links;
it is not a metering proxy for arbitrary external model calls. A production
model adapter and central call accounting remain separate integration work.

## Evaluation boundary

Selection tools do not accept scoring references. After saving decisions, run:

```bash
uv run --locked --extra canonical python -m experiments.canonicalization.evaluate \
  scoring-references.json run/results run/evaluation.json
```

The evaluator reports errors against audited dataset coordinates. This is not
independent real-world semantic ground truth. Keep development and holdout
objects separate, count repeated rotations as correlated observations, and do
not replace original failed decisions after inspecting their errors.

For repeated input rotations, use `experiments.canonicalization.consistency`
after saving every decision. It compares recovered frames as `R @ Q`, separates
up-only and full-frame agreement, and does not count abstention as a successful
orientation recovery. See the [rotation-consistency experiment](experiments/2026-10-27-rotation-consistency.md)
for the frozen protocol, invocation and measured limitations.

Geometry-only candidates have limits: support poses depend on a uniform-density
convex hull; surface PCA can miss the true functional direction; scans can lose
the parts needed to distinguish upright signs or front. The agent should request
additional observations only after available evidence fails to resolve those
questions, rather than forcing every object into a complete frame.
