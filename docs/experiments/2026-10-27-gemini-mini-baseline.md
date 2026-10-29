# Mini v1: partial live baseline and full geometry candidate coverage

The fixed geometry + Gemini worker was tested on October 27, 2026, without
changing category rules, prompts, candidate generation or model settings. The live
run attempted **23 of 60 planned rotation cases**, covering all **12 ready NOCS
objects**, before stopping after two consecutive API transport timeouts. It did
not complete the whole 37-object catalog. A separate offline diagnostic evaluated
the geometry candidates for all 60 prepared inputs.

The main finding is that a useful candidate existed in every prepared case at the
15-degree threshold, but the visual selector did not always choose it. One laptop
produced nearly opposite full frames across two input rotations. This is evidence
against treating a successful export or a plausible preview as semantic certainty.

## Fixed protocol

- Existing `pilot-semantic-v1` rules: +Y up and +Z front where defined.
- Surface PCA and convex-hull support planes for up; four horizontal PCA directions
  per selected up for front. No OA V2 or PointSO calls.
- Gemini native API, requested and returned model `gemini-3.8-flash`.
- Uniform-gray NOCS GLBs; no textures or supplemental real photographs.
- Maximum two calls per axis, four per case, with the existing 60-second socket
  timeout. No failed case was retried or replaced after scoring.
- One preselected rotation per category/partition first, then remaining rotations.
  The original input index contains five rotations for each object.
- Original development/holdout labels are retained. These objects have appeared
  in earlier project diagnostics; this is not a fresh independent holdout claim.
- Reference coordinates and applied rotations stayed outside inference. Scoring
  used the separately audited NOCS convention after the live run stopped.
- Thresholds are 5, 10 and 15 degrees; they are diagnostic thresholds, not calibrated
  automatic dataset-admission rules. Every export retains `not_granted` approval.

The user confirmed the Gemini project was Free Tier with billing disabled. No
billing changes or model/provider fallback occurred. The first transport failure
stopped the initial driver; a recorded continuation ran only previously unstarted
cases with identical worker settings. It stopped after two consecutive transport
failures. No HTTP 429 was returned; the cause of the transport timeouts was not
established. The 60-second socket limit is not a total job deadline.

## Dataset readiness

| Group | Selected objects | Ready for this fixed worker | Missing prerequisites |
| --- | ---: | ---: | --- |
| NOCS | 12 | 12 | None for the shape-only diagnostic |
| OO3D-9D | 20 | 0 | All 20 meshes absent; 16 instances also lack supported category rules |
| Own captures | 5 | 0 | Four meshes absent; four categories unsupported, with overlap |

The existing own-capture hole-punch mesh is preserved, but `hole_punch` is not a
supported convention in this worker. Unsupported cases were not silently relabeled
or counted as inference failures. All 25 unavailable objects are listed separately
in the local availability report.

## Live worker outcomes

| Outcome | Rotation cases |
| --- | ---: |
| Up-only proposal, for a convention with unspecified yaw | 10 |
| Full-frame proposal | 6 |
| Explicitly unresolved after extra views | 2 |
| API transport failure | 5 |
| Not started after the stop condition | 37 |

There were **33 counted requests**, including five without saved responses, and
28 validated responses. Median attempted-case time was **34.16 seconds**, with a
range of **11.15–110.74 seconds**, including local rendering and cloud waits.
Recorded usage from returned responses was 403,815 total tokens. This is not a
billing receipt and excludes unknown usage for timed-out requests.

No human selected object directions during this run. Unresolved cases represent
missing evidence, not completed human confirmations or a measured app intervention
rate; the worker has no contributor-facing confirmation UI.

## Accuracy against audited dataset coordinates

Errors below describe returned proposals. Failure and abstention counts remain
visible rather than being silently dropped from the denominator.

| Metric | Eligible planned / attempted | Returned proposals | Within 5° | Within 10° | Within 15° | Median among proposals |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Up direction | 60 / 23 | 16 | 15 | 15 | 15 | 0.89° |
| Full frame | 30 / 11 | 6 | 2 | 4 | 5 | 6.04° |

Thus the observed count of correct up proposals at 15 degrees is **15/23 attempted
cases**, not 15/16 as an end-to-end completion rate. For full-frame cases it is
**5/11 attempted**, not 5/6. These are correlated rotations of a small set of
objects, not independent samples or conclusive population estimates.

The mean full-frame error among proposals was **34.22°**, much higher than its
median because one laptop result had **48.99° up error and 178.87° full-frame
error**. A second rotation of that same laptop yielded 1.57° up and 2.30° frame
error. Their recovered frames differed by **179.93°**. The returned semantic
rationales sounded plausible in both runs, so rationale fluency is not a reliable
correctness signal.

Five other objects with two returned up proposals had pairwise up differences
below 0.00002°. This is limited two-rotation evidence; no object completed all five
live rotations. The failed and unstarted cases prevent a full consistency claim.

## Geometry candidate coverage: no API required

After inference, an oracle evaluator compared all geometry-generated candidates
with the audited reference. This diagnostic measures whether a good choice was
available, not whether the model could find it. For full frames it enumerated the
four front candidates for **every** up candidate; this is broader than the live
worker, which commits to an up before generating its front candidates.

| Candidate set | Cases | Best candidate within 5° | Within 10° | Within 15° | Worst best-candidate error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Up | 60 | 60 | 60 | 60 | 3.20° |
| Full frame | 30 | 15 | 25 | 30 | 10.33° |

The up generator produced 8–12 alternatives per input. All 2,015 cached rotation
matrices passed finite SO(3) validation. Trimesh emitted a square-root warning
while computing stable poses; no nonfinite candidate rotations were exported.
Physical stability still rests on the uniform-density convex-hull approximation.

On this small prepared set, candidate coverage supports retaining the geometry
baseline. It does not establish reliability for arbitrary real objects. The
full-frame results also expose a precision limit: half the cases have no candidate
within 5 degrees, even for an oracle selector. Better semantic selection and finer
orientation proposals address different failure modes.

## Appearance and the next controlled comparison

The renderer preserves imported GLB materials and colors; this NOCS track lacks
appearance because the source meshes do not include the referenced texture assets.
The worker already supports identity-bound supplemental photographs through
`--observations`, with explicit correspondence evidence. Photos were deliberately
not supplied in this fixed run.

Inspection of the existing SAM 3D hole-punch GLB found `POSITION` and `COLOR_0`
attributes, but no material, texture or embedded image entries. It contains
generated vertex colors, not a UV image texture or an exact copy of the capture.
Preserve original user photographs as independent evidence for logos, text and
small parts, even when a generated mesh has colors.

A future comparison should hold object identities and direction definitions fixed
while adding textured renderings or identity-matched real photos. For a circular
bottle, geometry-only PCA directions need not align with the main label; appearance
may need to propose/refine yaw as well as help select candidates. A label-facing
bottle track needs reviewed front references: the current NOCS bottle metric
ignores yaw and cannot measure that improvement.

## Artifacts and verification

Local artifacts are under `data/experiments/runs/gemini-mini-baseline-01/`:

- `protocol.json`, `analysis-plan.json`, `availability.json`: scope, versions,
  thresholds and readiness; credentials are excluded.
- `jobs/`: raw replies, selected decisions, stage hashes and original failures.
- `geometry-candidates/`: reference-free cached candidates for all 60 inputs.
- `evaluation.json`: reference scores, conditional front oracle and pairwise
  consistency; [versioned result snapshot](2026-10-27-gemini-mini-baseline-results.json).
- `report.html`, `overview.jpg`, `previews/`: first planned rotation per object,
  with identical before/after camera presets, plus a table of all 60 cases.
- `execution-summary.json`, `transport-continuation.json`: counted requests,
  timing and the one recorded continuation of unstarted cases.
- `run_batch.py`, `continue_batch.py`, `prepare_oracle.py`, `score_batch.py`,
  `preview_batch.py`, `make_report.py`: local experiment orchestration. Worker
  implementation and inference conventions were unchanged.

An initial offline scorer invocation failed when passing JSON lists to the existing
NumPy angular-error helper. The original scorer and its checksum are preserved;
the repair only converts lists to arrays. It did not change metrics, thresholds,
reference labels or model outputs.

Ruff check, Ruff format check and mypy passed; **152 pytest tests passed**. These
software checks do not establish model accuracy. All 23 executed up-candidate sets
matched the offline cache; the six returned full-frame errors were independently
checked with SciPy rotation magnitudes. The figure was visually inspected.
Do not silently resume this stopped experiment after inspecting errors or overwrite
its failures; a follow-up run should retain this baseline and disclose its changes.
