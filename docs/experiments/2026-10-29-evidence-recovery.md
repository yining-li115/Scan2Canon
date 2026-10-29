# Shared-frame evidence recovery diagnostic

This follow-up preserves [trial 01](2026-10-29-shared-frame.md) and reruns the same
two mugs with the same stored shared-frame proposal and saved OA predictions.
It is development after observing failures, not an independent benchmark.

## Changes

- Separate observation requests from final-pose validation. Invalid interim
  direction references are logged as deferred diagnostics; a valid bounded tool
  can still obtain evidence. Final exports retain all original gates.
- Keep a per-job uncertainty floor for the same image bytes, candidate and part
  relation. The policy cannot narrow its own previous radius merely by changing
  points or IDs. This does not independently validate localization.
- Add presentation-only XYZ overlays using saved cameras, glTF basis transforms
  and per-candidate rotations. Unresolved axes use dashed lines and question marks.
  Raw inference images remain unchanged.

No category rules, new service, GPU dependency or arbitrary rotation action was
added. The finite OA candidate pool and the one-refinement limit are unchanged.

## Protocol and results

Each object used a fresh job with a four-generation budget and Claude
`claude-sonnet-5-5`. Both received existing colored meshes, source renders, original
photos and the exact pair-derived +Y base-to-opening / +Z body-to-handle definition.

| Object | Executed tools | Outcome |
| --- | --- | --- |
| Blue mug | `refine_up`, `render_view` right | Failed; no canonical export |
| Red mug | `refine_up`, `render_view` left | Up-only, candidate 12 |

Both models left interim measurements empty, as allowed by the revised prompt.
Supplemental views executed without the reference errors seen in trial 01. Thus
this run exercises the simplified valid path; the deferred-invalid branch is
covered by a regression test, not claimed as a live event in this run.

Blue selected candidate 13 on turn 3, but its projected up check was insufficient.
On turn 4 it changed to candidate 12 and omitted the required parent candidate 4
from its citations. The export was rejected. It also changed image/candidate, so
the same-image uncertainty-floor safeguard was not exercised by this real case.
Do not reuse trial 01's blue export as if it came from this run.

Red's turn-3 selected candidate 12 had a visible handle pointing opposite its +Z
axis. The reply also omitted the required parent citation. The last turn retained
the candidate's supported up, cited the parent and explicitly abandoned front.
The final 2D up check was 0.31 degrees with conditional interval 0–11.90 degrees;
this is not a 3D pose error. The available rendered evidence now exposes the yaw
problem instead of leaving it merely hidden behind the body.

**Both complete canonical frames are still not established.** Better tool execution
did not yet improve the complete-frame success rate. Next development should make
candidate/parent bookkeeping easier and provide bounded, provenance-preserving
yaw alternatives when the agent has evidence that the selected front has the
wrong sign. Those alternatives must be rendered and checked, not silently applied.
Independent landmark/reference evaluation remains necessary before scaling.

## Local artifacts and verification

`data/experiments/runs/two-mug-alignment-02/` contains:

- `audit.json`, `review.py`: saved outcomes, errors, tool history and actual usage.
- `diagnostic-axes.png`: blue rejected candidates and red partial export, all with
  XYZ axes. These are diagnostic views, not two accepted final canonical meshes.
- `blue-mug/rejected-candidate-axes/`: annotated right-side candidate renders;
  Pose 00/01/02 correspond to candidates 12/13/14. None is an accepted export.
- `red-mug/axes/`: six camera views of the up-only mesh; X/Z remain unresolved.
- Each `job/`: raw calls, checksummed stages, direction checks, trace and state.

Eight generation replies used 213,953 input and 18,255 output tokens. At the
previously verified $2/M input and $10/M output rates, estimated usage is
$0.610456. This is not a live account-balance read. No billing settings changed;
no new GPU inference or manual pose/landmark edits occurred.

Ruff and mypy passed, and the full suite passed all 266 tests.
Regression tests verify that an invalid interim measurement does not block
a valid observation, the same error cannot authorize a final export, replay avoids
new calls, shrinking same-image uncertainty cannot authorize export, perspective
and glTF axes are correct, and overlays preserve raw evidence and scope.
