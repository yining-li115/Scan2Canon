# Two-mug shared frame development trial

## Question and protocol

Can two existing, visually different mugs agree on observable direction meanings
without a manually authored category table, then apply those meanings through the
current evidence-agent gates? This is a development diagnostic using previously
examined objects, not a held-out classification or pose benchmark.

The pair task used the two imported Claude assessments documented in
[object library](../object-library.md). It received six source renders and two
original photos per instance, plus previous unverified direction hypotheses.
It did not receive display names, previous identity labels or selected poses.
Direction descriptions still expose semantic clues; this is not an independent
recognition evaluation.

- Model: `claude-sonnet-5-5`, same bounded Claude adapter as the previous trial.
- Pair task: maximum two calls; one generation used.
- Alignment: maximum four calls per object; eight generations used in total.
- Compute: existing meshes and cached OA V2 predictions; local geometry and Blender.
  No new reconstruction or GPU model inference.
- No manual pose changes, landmark edits, admission, category table or extra calls
  after the original budgets were exhausted.

## Shared proposal

The model recognized the two objects as handled drinking mugs and proposed:

| Axis | Exact stored part relation |
| --- | --- |
| +Y up | `closed flat base bottom` → `open rim / cavity opening` |
| +Z front | `body center` → `side loop handle (grip side)` |

Front is an explicit handle-facing task convention, not an intrinsic or universal
front. The proposal retains graphic-facing and handle-opposite alternatives. It
describes applicability to open-top vessels with a closed base and a single side
loop handle. The applicability prose also mentions omitting front for missing
handles while elsewhere excluding handleless vessels; it has not been reviewed
into an operational admission rule. Do not apply it automatically beyond this pair.

Stored proposal: `0ba5ea98-15e9-522d-92a3-a29329154d76`, state `proposed`, definition
version 1. Pair artifact: `data/experiments/runs/two-mug-shared-frame-01/result.json`.
The original two up-only assessments remain immutable.

## Alignment results

| Instance | Executed evidence tools | Final outcome | What remains |
| --- | --- | --- | --- |
| Blue mug | Support-plane up refinement; right-side renders | Full-frame proposal, candidate 14 | Independent localization and 3D pose validation |
| Red-handled mug | Support-plane up refinement | Up-only proposal, candidate 12 | Handle-facing yaw not established |

Blue: the third reply omitted the refined candidate's original parent citation.
Its front check was also indeterminate: estimated 2D deviation 9.16°, conditional
interval 0–23.92°. The fourth reply cited the parent, changed the measured points
and reduced the uncertainty radius from 0.04 to 0.015 on the **same image**. That
yielded 10.12°, interval 4.07–16.18°, passing the existing 20° gate. No new visual
evidence independently justified the narrower uncertainty. This is a concrete
limitation of policy-supplied landmarks and uncertainty, not evidence that the
3D pose error improved. The up check was compatible at 0° projected deviation.

Red: two replies contained invalid image/candidate references, which the harness
rejected. A proposed additional render therefore never executed. The last reply
abstained on front within the remaining budget and exported only up. Its projected
up deviation was 0.13°, conditional interval 0–13.78°. These are 2D consistency
numbers, not 3D pose accuracy or comparisons with published SOTA.

The final preview confirms why this cannot be called two successful full frames:
the blue handle faces +Z, while the red handle remains on the other side and its
yaw is explicitly unresolved. The harness did not rotate the red mesh manually
to make the examples agree.

## Artifacts and cost

All results are local under `data/experiments/runs/two-mug-alignment-01/`:

- `audit.json`: outcomes, source hashes, actual usage, checks and failures.
- `canonical-results.png`: identical camera views of both exports; coordinate-view
  labels are not semantic validation labels.
- `landmark-checks.png`: actual final points, uncertainty and projected axes.
- `<case>/job/trace.md`, `events.jsonl`, `calls/`, `direction-checks/`: complete logs.
- `<case>/job/work/agent-export-4-1/canonical.glb` and `canonicalization.json`:
  preserved output and provenance.
- `<case>/canonical-preview-02/`: successful post-inference Blender previews.

The first sandboxed preview attempt crashed in Blender before rendering. Its
failed `blue-mug/canonical-preview/` artifacts remain; a separate local attempt
outside the sandbox succeeded. This did not rerun any model decisions.

Nine generation replies used 218,881 input and 20,889 output tokens, with no cached
tokens. At the previously checked $2/M input and $10/M output rates, the estimate
is **$0.646652**. This is a usage-based estimate, not a fresh account-balance read.
No recharge, payment method or billing setting was changed.

## Conclusion and next experiment

The storage-to-pair-proposal-to-instance-execution loop now runs. It demonstrates
agreement on a task meaning and preserves one partial outcome; it does **not**
establish same-class full-frame consistency or automatic classification at scale.

Before scaling, improve the generic evidence action contract so requesting another
view does not require unrelated final-pose measurements, and make landmark evidence
more independent from the candidate selector. In particular, shrinking a model's
own uncertainty on unchanged evidence cannot serve as independent confirmation.
Keep the current run unchanged as the development baseline. Then evaluate on fresh
views/rotations and unseen functional matches and lookalikes, with independent
references and separately reported abstention rates.

Validation for the implementation: Ruff passed, mypy passed for `src`, `experiments`
and `tests`, and all 261 pytest tests passed. New tests cover pair correction,
immutable/idempotent recovery, corrupted evidence, uncertain transport dispatch,
no forced shared frame, exact shared-meaning enforcement and wrong-mesh rejection.
