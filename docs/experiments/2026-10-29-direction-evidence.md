# Structured direction evidence and projected part checks

This development trial extends the table-free workflow with explicit direction
hypotheses, separate uncertainty reasons, unverified symmetry hypotheses and a
deterministic 2D relation check. See the
[agent contract and limitations](../canonicalization-evidence-agent.md#direction-meanings-and-projected-part-checks).

## Frozen protocol

Use the same three inputs as `no-conventions-01`: blue faceted mug, red-handle
patterned mug and spray bottle. Reuse saved colored meshes, OA predictions and
original photographs without supplying earlier decisions to the policy.

The configured policy is Gemini 3.8 Flash on the user's Free-tier project.
Each fresh job has a maximum of four provider calls, including schema repair
and any existing model fallback, for twelve calls total. Tools and rendering
run on the local Mac. No new GPU or reconstruction inference is needed.
Code versions and input hashes are frozen in
`data/experiments/runs/direction-evidence-01/protocol.json`.

Before model inference, the implementation passed Ruff, mypy and all 237 tests.
Tests include known pinhole projections at off-center landmarks, reverse
directions, uncertain/degenerate projections, unregistered-photo refusal,
explicit competing definitions, numeric-conflict feedback and bounded recovery.
Tests use declared synthetic measurements; they are not semantic accuracy tests.

No pose is manually corrected, no category direction definitions are introduced,
and no model-estimated point is replaced by a hand-picked point. The output may
remain partial or unresolved. A successful export cannot count as ground-truth
semantic validation. These are development examples without held-out pose labels.

## Results

The first batch, `direction-evidence-01`, dispatched one call for each of the
three cases. All three were rejected with HTTP 400 before model output. One
additional diagnostic request returned `INVALID_ARGUMENT` with the message
"Request contains an invalid argument." No pose result exists from those four
requests; they remain counted.

The Gemini transport schema was then simplified following
[Google's documented guidance](https://ai.google.dev/gemini-api/docs/structured-output?hl=en).
Numeric and array-length bounds are omitted from the provider's grammar, while
the complete local Pydantic schema still rejects out-of-bounds replies before
actions or exports. Types, field structure, required fields and enums remain.
The error did not identify the offending keyword; the exact compiler constraint
is not known. An additional test covers preserved local validation.

A fresh batch under `direction-evidence-02` runs blue mug and spray bottle only,
with at most four calls each and a stop-on-run-failure guard. Across initial
failures, diagnosis and this batch, the maximum remains twelve provider calls.
The first fresh request was accepted and selected support-plane refinement.
The revised protocol records its own code hashes. These examples were chosen to
exercise structural/graphic and functional/presentation alternatives; they are
not a representative success-rate estimate.

The second batch completed in four calls, leaving four requests within the
overall budget. A final fresh red-handle mug job under `direction-evidence-03`
uses at most that remaining budget, with unchanged code and input evidence.

## Completed checks

| Case | Model calls | Export | Selected up relation | Projected discrepancy |
| --- | --- | --- | --- | --- |
| Blue mug | 2 | up_only | Resting base → open rim | 0.00° |
| Spray bottle | 2 | up_only | Bottom base → top of dispenser head | 0.54° |
| Red-handle mug | 2 | up_only | Resting base → open rim | 0.13° |

All three agents selected `refine_up` followed by `finish`. Their measured points
passed the deterministic projection check. No rejected reply, fallback or user
capture occurred in these three successful jobs. All identify asymmetric parts
as counterarguments to overall yaw symmetry, while retaining multiple front
definitions. These symmetry statements remain unverified policy hypotheses.

The discrepancies are **2D agreement between model-estimated points and the
candidate axis**, not 3D pose error or independent semantic accuracy. For the
blue mug the uncertainty-conditioned interval is 0–5.14°; for the bottle it is
0–3.77°. Both depend on a model-supplied 0.02 normalized location radius. These
are not measured confidence intervals. Post-inference overlay inspection places
the endpoints on the visible bottom/top contours in these examples, but no
independent landmark annotation or multi-view triangulation was performed.

The blue mug's alternative fronts are opposing-handle and graphic-facing
presentation meanings. The bottle keeps functional spray direction and label
presentation direction separate. This makes the source of ambiguity inspectable;
it does not choose a project-wide meaning or establish cross-instance alignment.
The bottle's detailed functional description is still a model hypothesis rather
than a verified functional-part relation.

The red-handle mug retains graphic-facing presentation and handle-relative
drinking directions. Its selected candidate 12 is refined from parent 1 by
6.41° according to the support-plane tool. The agent also supplied a measurement
on parent candidate 1: **both the refined and unrefined poses pass the 2D check**
(0.13° and 0.47° projected discrepancy respectively). The original's measured
interval is 0–9.57° with a stated 0.03 location radius. This is a concrete example
of why passing a single projection does not rule out depth-direction tilt,
point-localization error or inappropriate part definitions. The support correction
itself is not ground-truth pose error. The actual comparison overlays are preserved;
the original candidate is not relabelled as correctly oriented.

The final total is ten dispatched provider requests: three initial HTTP 400
failures, one HTTP 400 diagnostic and six successful calls across three fresh
jobs. No further calls were made. All 238 tests, Ruff check/format and mypy passed
after the transport correction. No GPU or user confirmation was used, but all
outputs remain partial and none is approved for dataset admission.

## Inspect locally and next limitation

- `direction-evidence-01`: failed first batch, frozen protocol and diagnostic count.
- `direction-evidence-02`: successful blue mug and spray bottle jobs,
  `landmark-checks.png` (all three objects plus the red mug's original comparison),
  and `visual-review.json` with plotted artifact hashes.
- `direction-evidence-03`: successful red-handle mug job and its own frozen protocol.
- Each job contains `direction-checks/agent-N.json` and `agent-N/check-*.png`,
  `calls/`, `events.jsonl`, `trace.md`, and the selected GLB/manifest under `work/`.

The implemented check is deliberately named projected-part compatibility. A
follow-up should require consistent observations of the same semantic parts from
multiple registered render cameras and examine their sensitivity to localization
uncertainty. Those observations still need valid cross-view correspondences;
simply repeating a model's preferred points is not independent verification.
Selecting a common axis meaning across instances remains a separate open problem.

All failures and raw replies remain preserved. Nothing was manually rotated or
relabelled after the model decisions. Full semantic orientation is still open.
