# Reviewing the learned OA reference

This development trial keeps OA V2 as the orientation backbone and changes the
agent's task from independently choosing a useful front to reviewing the model's
proposed frame. It uses the same two mugs as the preceding
[distribution-fusion trial](2026-10-29-distribution-fusion.md), not held-out objects.

## What the sources establish

The [OA V2 paper, section 4.2](https://arxiv.org/html/2601.05573v1) describes
multiview pseudo-annotation and periodic distribution fitting. Its category-level
calibration checks symmetry types. This does not establish a textual rule fixing
the mug handle to a particular positive axis. The
[official project](https://orient-anythingv2.github.io/) uses X as front, Y as left
and Z as up; our proper change of basis maps these to Z front, X left and Y up.

We did not verify mug-specific training annotations. The public training-data
listing contains large render archives; inspecting that listing is not an audit
of the annotations inside. The implementation therefore records
`category_annotation_verified: false`. It must not invent a handle/graphic rule
and attribute that rule to OA.

## Implementation

The v4 evidence agent receives a versioned `orientation_reference` in every
request and preserves it in the final report. A `model_reference` hypothesis
describes observable part relations associated with an OA branch. Other functional
or presentation hypotheses may remain unselected without automatically vetoing
that branch. Missing evidence, genuine competing OA branches and contradictory
observations still block the affected axis.

No projection tolerance, uncertainty floor, support-inspection requirement,
candidate-membership check or call budget was relaxed. A full-frame result still
needs compatible part measurements for both axes. Those estimates come from the
same model and do not independently verify semantic correctness.

Completed v4 jobs can be imported into the object library with their reference
provenance. Historical v3 imports remain supported without retroactive reference
assignment. No category definition or approved dataset entry is created.

## Protocol and first attempt

Both jobs use the original twelve spherical mesh renders, matching saved OA
logits, six distribution-fusion candidates, colored meshes and original-photo
bundles. No new OA/GPU inference occurs. Each job permits at most four Claude
calls, with local Blender tools and unchanged geometric checks.

The first attempt is preserved at
`data/experiments/runs/oa-reference-review-01/`. Both jobs executed `refine_up`
but exhausted their budgets without an export. The blue job first omitted
`directions.measurements`. Subsequent blue and red replies proposed an up-only
export while setting `semantics.status` to uncertain because front was uncertain.
The contract requires supported identity/up for any export and represents front
uncertainty separately. Generic schema feedback did not identify that conflict.

The second attempt uses a fresh directory,
`data/experiments/runs/oa-reference-review-02/`. It clarifies identity versus front
status and adds bounded field/type validation feedback, without echoing model
values or unknown keys. The specific identity/pose conflict explains both legal
outcomes: supported identity/up can retain unknown front; uncertain identity must
remain unresolved. No failed reply is rewritten or accepted retroactively.

Both directories retain a frozen protocol, source snapshot, raw calls, checks,
events and readable trace. These changes followed observed failures, so the
second attempt is a development rerun, not an independent evaluation.

## Actual second-attempt results

| Object | Calls | Executed evidence actions | Final proposal |
| --- | ---: | --- | --- |
| Blue mug | 2 | `refine_up` | Candidate 6, `up_only` |
| Red mug | 4 | `refine_up`, `render_view` from above | Candidate 6, `up_only` |

The red job's third reply omitted its refined candidate's original parent citation.
The harness rejected that choice; the fourth reply supplied the parent comparison
and passed. No extra calls or post-hoc edits to the model choice were used.
Both exported up axes passed projected-part compatibility, using model-estimated
landmarks. These checks are not measured 3D orientation error.

Both outputs are byte-identical GLBs to the preceding distribution-fusion run.
The code and evidence behavior changed, but the geometry and accepted pose scope
did **not** improve. Both agents still abstained on yaw: they saw handle-away and
handle-toward branches without enough evidence to assign one the learned front
sign. The blue branch score gap between candidates 0 and 3 is 0.12881; the red gap
between candidates 0 and 4 is 0.21992. These are uncalibrated ranking units, not
probabilities or evidence that the alternatives are equally credible.

This exposes a remaining distinction: observing which side contains a handle
does not independently reveal the training frame's intended sign. Merely including
a lower-ranked opposite candidate also does not demonstrate that the leading
prediction is wrong. The present agent still asks for evidence it cannot obtain
from these images alone. More rendering cannot by itself audit the training convention.

The next evaluation should distinguish an OA-ranked pose **proposal** from an
independently checked pose: measure leading-branch stability across views and
instances, retain model ranking and counterevidence, and calibrate abstention on
independent annotations. Repeatedly changing the prompt until it accepts these
two examples would not establish that capability.

Final local artifacts include `summary.json`, each job's `trace.md`, both
`canonical.glb` files, all seven axis views per object, and `final-pair-axes.png`
with +Z, +X and +Y views. Solid green Y denotes proposed up; dashed red X and blue Z
retain the unresolved yaw. No dataset admission occurs.

Validation: Ruff and mypy (`src tests experiments`) pass; all 290 tests pass.
New checks cover reference propagation, storage provenance, rejection of changed
reference authority, preserved export gates and recoverable schema diagnostics.
