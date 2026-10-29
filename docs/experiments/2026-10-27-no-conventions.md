# Canonicalization without a category direction table

The evidence agent no longer has a category direction table or a
`lookup_convention` action. This experiment examines what it chooses from
existing colored mesh renders, OA proposals and original photographs. It does
not add a replacement taxonomy, direction registry or category-specific prompt.

## Code and migration

- Delete `experiments/canonicalization/conventions.json`.
- Remove the lookup action from the structured output contract, tool dispatcher
  and policy context. The current route is `oa-evidence-agent-v2`; old jobs cannot
  resume under this route.
- Remove the same table dependency from the older staged geometry workflow.
  Category text is an unverified identity hypothesis, not an allowlist or a front
  definition. Staged decisions use schema version 2 and bind to evidence hashes.
- Preserve fixed coordinate notation (+Y up, +Z proposed front), rigid-transform
  checks, evidence validation, call budgets, checkpoint integrity and the review
  of support-plane refinements. These are not per-category direction definitions.
- Update the project skill and active documentation. Historical experiment
  records remain unchanged and still describe their original table-based runs.

## Protocol

Outputs are under `data/experiments/runs/no-conventions-01/`.
`protocol.json` records input hashes, code versions and the configured model.

Three fresh jobs reuse unchanged meshes, OA predictions and original observations:

| Case | Existing input directory |
| --- | --- |
| Blue faceted mug | `own-appearance-01/mug` |
| White patterned mug with red handle | `mug-consistency-5654-01` |
| Spray bottle | `own-appearance-01/spray-bottle` |

Each job permits at most four provider calls, including fallback and validation
corrections. The total budget is twelve. The configured Gemini Free-tier policy
runs remotely; Blender tools and exports run on the local Mac. No GPU inference
or new reconstruction is required. Previous decisions, target categories and
reference poses are not supplied to the policy. No pose is manually corrected
after inference. Source views and saved OA candidates retain upstream priors;
removing this table does not make the experiment free of learned priors.

The prompt now explicitly allows partial orientation when multiple fronts are
defensible. Thus this is a development trial of the revised workflow, not a
single-variable benchmark. Earlier blue-mug and spray-bottle decisions used
Gemini 3.6, while the previous red-handle mug used Gemini 3.8. There is no
independent pose ground truth, repeated-run estimate or dataset approval.

## Results

All three jobs completed with Gemini 3.8 Flash, using seven provider calls in
total, with no fallback, rejected replies, manual choices or capture requests.
Every output is **up_only**, not a complete canonical frame.

| Case | Actions | Calls | Unresolved front |
| --- | --- | --- | --- |
| Blue mug | refine_up → finish | 2 | Handle-related orientations versus a graphic visible in the photograph but not retained clearly in the mesh |
| Red-handle mug | refine_up → render_view(top) → finish | 3 | Handle-relative orientation versus printed graphics |
| Spray bottle | refine_up → finish | 2 | Nozzle direction versus packaging display direction |

Selected candidates were 12, 13 and 12 respectively. Their support-plane up
corrections were 8.74°, 17.10° and 12.18°. These numbers are corrections from
their OA parents, **not angular errors against ground truth**. Post-inference
render inspection shows the mugs' openings and the bottle head upward, with the
resting bases horizontal. Export checks confirm unchanged stored appearance and
geometry under the saved rigid transforms. No semantic ground-truth check exists.

The model used the original photographs and appearance-bearing renders, rather
than only shape. However, citing a photo does not verify its claimed mesh
correspondence. Its estimated angle between the bottle label and nozzle is an
unverified visual description, not a computed measurement.

## Problems exposed

1. **Instance recognition does not establish a common frame.** Both mugs are
   identified as drinking mugs, but neither receives a unique horizontal direction.
   The representative exported yaws must not be counted as a cross-instance
   consistency success, even if the handles appear roughly aligned in a plot.
2. **Missing observations and competing definitions are different problems.**
   More photos may identify a nozzle or graphic, but cannot by themselves decide
   whether the intended canonical axis should mean function or presentation.
   The current output preserves these reasons mainly as prose rather than
   separately testable claims.
3. **Semantic ambiguity is not geometric symmetry.** The red-mug response refers
   to cylindrical symmetry despite a handle and graphics. These break continuous
   yaw symmetry. `up_only` records an unresolved direction, not a certified set
   of symmetry-equivalent rotations. The current harness checks contract and
   geometry integrity, but does not detect this reasoning error.
4. **Photo-to-mesh evidence remains weak.** The blue-mug response notices a photo
   graphic absent from the mesh and stops at a partial frame. This does not prove
   that the photo cannot constrain orientation through other shared parts. No
   registered photograph pose or verified part correspondence was available.
5. **Conservative completion is not full task coverage.** Zero user interruptions
   here comes with zero full-frame proposals. It must not be reported as three
   successful complete canonicalizations. Repeated runs and independent labels
   are still needed to measure accuracy, abstention and consistency.

These observations motivate a later object-independent representation of part
evidence and competing axis meanings, followed by measurable cross-view and
cross-instance checks. This trial does not introduce that solution or another
hand-authored category table.

## Inspect and reproduce

Under `data/experiments/runs/no-conventions-01/`:

- `run.py`, `protocol.json`, `summary.json`: batch inputs, frozen versions, model
  identity, budgets, durations and outcomes. Running the script again against the
  existing directories will not overwrite the jobs; use fresh output paths for
  a new experiment.
- `canonical-results.png`: oblique and top views of the actual exported poses.
  Green is +Y up; dashed red/blue axes show representative X/Z only. They are not
  selected semantic front directions. Meshes are independently framed, not scaled
  for metric comparison.
- `<case>/job/trace.md`, `events.jsonl`, `calls/`: actual action histories, raw
  model replies and attached evidence references.
- `<case>/job/work/agent-export-*/canonicalization.json` and `canonical.glb`:
  selected transforms, semantics, preserved mesh appearance and integrity checks.
- `visualization-provenance.json`: plotted result hashes and no-manual-pose-change
  declaration. Visual review is separate from the model's decisions.

The final legacy staged prompt was also clarified after the batch to list
`no_unique_front` consistently among allowed front responses. That helper is not
used by the evidence-agent trials. Their original code hashes remain in the
protocol; completed jobs are preserved rather than resumed under changed code.

Validation: Ruff check/format and mypy passed; all 229 pytest tests passed. The
suite covers table-free unknown identities, partial front decisions,
removed-action rejection, old-route resume rejection, bounded recovery and rigid
export integrity. The project skill passed its skill-creator validation as well.
