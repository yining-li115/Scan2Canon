# Official OA parity and bounded up refinement

This development experiment addresses visible tilt in the own-capture mug and spray
bottle. Inputs and previous outputs remain unchanged under
`data/experiments/runs/own-appearance-01/`. New artifacts are under
`data/experiments/runs/own-up-refinement-01/`.

## Official inference and coordinate checks

The existing isolated A100 OA environment reran all twelve views per object with
`experiments.orientation.infer --verify-official`. For every view, the adapter's
preprocessing/forward/decoding agreed with the upstream `inf_single_case` function.
Both parity reports say `passed`. New predictions were identical to the saved
predictions and the maximum absolute saved-logit difference was zero for both
objects. This checks the same prepared renders, not model accuracy on real photos
or different background-removal settings.

The local Blender calibration also passed against the actual upstream
`axis_render.blend`: camera basis, named axes, Euler order and inverse pose recovery.
The tilted OA estimates therefore survive these integration checks. No calibration
offset was subtracted to make the outputs appear upright.

## Refinement boundary

`experiments/canonicalization/up_refinement.py` uses the existing stable-support
pose computation and actual mesh vertices near each proposed support plane. It
returns guarded alternatives without changing the mesh. A proposal requires:

- At most 25 degrees change from the parent pose.
- A near-contact polygon covering at least 10% of the projected footprint, using
  a band of 0.3% of the mesh bounding-box diagonal.
- No closely competing eligible support direction under the recorded angle guard.

These values are development guards, not calibrated probabilities or accuracy
thresholds. Convex-hull uniform density is a proxy, not real mass. Projecting the
parent front onto the corrected horizontal plane preserves its semantic hypothesis;
it does not establish a new front or resolve unknown yaw. The agent must inspect
the new candidate and compare it with its parent before choosing it.

The `refine_up` action accepts exactly three original candidate IDs. At most one
such action executes per job, adding at most three candidates. Failed or refused
corrections remain recorded. Tool recovery reuses saved model replies and restores
the checksummed candidate additions. The tool cannot modify original inputs,
replace a parent, raise the model budget or approve dataset admission.

## First live trial: optional tool was skipped

Both objects used two Gemini 3.6 Flash calls: convention lookup, then finish. Neither
called `refine_up`; both selected existing OA poses. These results remain preserved
under `mug/` and `spray-bottle/`. Merely listing the tool did not make the policy
detect and correct the tilt.

A subsequent harness change adds a recorded `support-assessment` stage before
model dispatch. Original candidates with an available correction above 3 degrees
require a refinement inspection before export. The agent can still choose the
original after citing its refined alternative, so a physical resting pose never
automatically overrides semantic evidence. Unsupported geometry does not fabricate
a support plane. A second live trial under `guarded/` permits two calls per object;
the two trials together allow at most eight calls. No baseline decision or pose
ground truth is supplied to either policy.

## Guarded trial results

Both objects called `refine_up`, received three rendered alternatives, and selected
a refined candidate in their second response. There were eight actual provider
calls across both trials, with no additional calls for recovery or visualization.

| Object | Selected parent → refined ID | Selected-parent up correction | Previous exported frame → support normal | Refined frame → support normal |
| --- | --- | --- | --- | --- |
| Mug | 4 → 13 | 8.74° | 15.61° | <0.001° |
| Spray bottle | 1 → 12 | 12.18° | 12.18° | <0.001° |

The mug selected a different OA parent from the previous export, so its old/new
comparison is not an isolated refinement ablation. Its projected front changed by
10.06°; the spray bottle retained the parent's projected front. Near-contact
polygon coverage changed from 2.60% to 32.21% for the selected mug parent and from
1.71% to 51.60% for the spray bottle. These are geometric diagnostics, not measured
stability or semantic ground-truth accuracy. Near-zero support residual follows
from aligning to the selected normal and is not an independent accuracy result.

The spray bottle completed its live export under
`guarded/spray-bottle/job/work/agent-export-2-1/`. Appearance and geometry checks
passed. Its label-front description agrees with the inspected result.

The mug's final response was initially rejected because `finish.expected_evidence`
was empty. This field now permits empty text for terminal actions, while actions
that obtain evidence must still describe what they expect. Pose, evidence and
budget checks remain in force. The failed original job is preserved.

An attempted fresh replay stopped when newly rendered PNG hashes differed; it is
preserved under `schema-revalidation/` and is not reported as a successful replay.
Instead, `revalidate-saved-evidence.py` verifies all original completed-stage file
hashes and both saved call hashes, reconstructs both original prompts with the exact
original images, and revalidates the unchanged replies. The only response-schema
delta is removal of the unconditional minimum length on `expected_evidence`.
The resulting diagnostic export is under `saved-evidence-revalidation/mug/`.
This used zero provider calls and is not a new autonomous model trial. The script
also runs the standard choice validation and geometry/appearance-preserving export.

Visual review found a separate semantic failure: the mug response says the handle
points toward +X, but its selected pose still places the handle approximately toward
+Z. Thus the new upright pose is useful, but the textual front description does not
match the selected frame. Neither object is approved for dataset admission. The
next semantic check should compare the declared functional landmark direction with
the selected mesh frame; refining up alone cannot solve that mismatch.

## Inspect the artifacts

Under `data/experiments/runs/own-up-refinement-01/`:

- `comparison/*-before-after.png`: previous export above, new proposal below;
  front and side cameras, with actual canonical axes projected from saved cameras.
- `comparison/*-after-four-views.png`: one fixed new frame from four cameras,
  not four independent orientation predictions.
- `comparison/metrics.json`: geometric diagnostics, front descriptions and report hashes.
- `guarded/*/job/events.jsonl`, `calls/` and `work/evidence-1-1/evidence.json`:
  decisions, raw replies, tool execution and refinement provenance.
- `saved-evidence-revalidation/mug/canonicalization.json`: explicit offline
  revalidation provenance and semantic review, separate from the failed live job.

The renderer retains vertex colors; these two meshes do not have UV image textures.
Refinement changes only the rigid pose, not shape or appearance.

## Validation boundary

Tests cover support recovery after arbitrary input rotations, refusal of large
updates and unsuitable support, preservation of unresolved yaw and parent front,
inspection before export, retaining the original after comparison, geometry/color
export, and interrupted-render recovery without redispatching the model.

Contact coverage and residual angle to the chosen support normal are internal
geometric diagnostics. They are not semantic accuracy against independent labels.
Original model replies, skipped refinements, validation failures and unsuccessful
attempts must remain visible in the experiment results.

Final local validation: Ruff check and format check passed, mypy passed for 110
source files, and all 227 tests passed using `uv run --locked --extra canonical`.
Additional terminal-action tests cover empty expected evidence on finish and reject
empty/whitespace evidence descriptions on nonterminal actions.
