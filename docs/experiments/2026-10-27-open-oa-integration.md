# Open semantic OA worker: implementation validation

Date: 2026-10-27. This record distinguishes mechanical integration from semantic
accuracy. The new route is described in [the worker guide](../open-canonicalization.md).

## Automated checks

Ruff lint and formatting, strict mypy, and all 176 tests passed. The 24 new tests
cover unknown-category hypotheses, correct OA-to-mesh coordinate composition,
symmetry/yaw restrictions, invalid decisions, bounded evidence requests, source
hashes, preserved model responses, original-photo correspondence requirements,
export recovery without repeated model calls, and both cloud request schemas.
Actual test GLBs preserve vertex colors and embedded texture/UV data after export.
Model answers and rendered test images are explicit fixtures, not learned results.

## Saved OA predictions with real Blender

Inputs:

- Render bundle: `data/experiments/inputs/video-5573-orientation-02`.
- Previously recorded real OA inference: `data/experiments/runs/orientation-video-5573-01`.
- Output: `data/experiments/runs/open-oa-integration-01`.

The new worker verified the saved mesh/render/model provenance, produced 12
candidates from the 36 symmetry-expanded hypotheses, rendered those poses in
Blender, and exported a rigid `up_only` proposal. The input predictions contained
both symmetry fits 2 and 4; neither was treated as an established physical label.
The omitted 24 hypotheses are recorded, rather than silently treated as rejected
by visual evidence.

The integration policy was scripted (`integration_fixture`,
`scripted-not-semantic-inference`). It selected candidate 0 only to exercise
export and recovery. This is not a claim that candidate 0 is upright or correct.
Inverse vertex recovery error was 3.67635e-9 against a 1.99825e-6 tolerance in mesh
units. A second run with `resume=True` reused the result without new model calls.
The contact sheet was inspected to confirm the object renders and appearance
were present; no accuracy label was assigned from that inspection.

No checkpoint was downloaded and no new GPU inference was performed in this
integration check. Original source captures and previous baseline results remain
unchanged. It does not establish class consistency, calibrated confidence, or
improved accuracy over the geometry baseline.

## First live Gemini attempt and correction

`data/experiments/runs/open-oa-gemini-01` used Gemini 3.8 Flash with the saved
hole-punch renders and OA predictions. No category label or scoring reference was
provided. The identity response described a two-hole paper punch and proposed up
and operating-front definitions. Its detailed functional claims remain model
hypotheses; they were not independently annotated.

The next response selected candidate 1, but cited `source:*` images from the
identity stage instead of the current `candidate:*` images. Local validation
rejected it and no mesh was exported. Both raw responses are preserved. This
failure exposed confusing cross-stage evidence in the pose prompt.

The corrected request carries the identity/appearance/parts descriptions without
the prior evidence IDs and reasoning. Its JSON schema enumerates only the current
image labels and available candidate IDs. Local validation remains mandatory even
with constrained generation. A regression assertion checks that source IDs are
absent from pose context and that schema labels match the actual supplied images.

The corrected live run, `data/experiments/runs/open-oa-gemini-02`, had a two-call
budget. Identity inference completed; the pose request encountered a Gemini
transport failure. The dispatched request remains counted in `state.json` and was
not retried. No pose response or canonical GLB was fabricated. Consequently the
corrected full cloud route has not yet completed a live validation, despite passing
all local tests. A future explicitly scheduled trial should use a new run directory;
`--resume` must not silently redispatch the uncertain network request.
