# Live appearance-aware rerun

This is a development rerun on the existing single-image SAM 3D hole-punch asset.
It uses stored vertex colors and the original reconstruction photo, not a UV texture
atlas or independently held-out photographs. There is no independent pose ground
truth for this object.

## Fresh OA inference

Twelve newly prepared known-camera colored views were uploaded to the existing
A100 40 GB server. Its clean OA runner at commit
`6dbe053f4cbab460fc25d0900ecb782b7dcf2c7f` matches the previously committed local
runner (SHA-256 `4f4461a4752815d53d735f2d73771497c1866b9e359aaad6fe32af04bd8f09e4`).
No remote source changes were needed for RGB inference. The current appearance
admission check ran locally before upload; the newer agent and rendering tools run
locally. The existing GPU adapter does not claim to have run the new strict flag.

New predictions are bound to the exact new render manifest, pinned checkpoint and
source revision. Results, logits, checkpoint provenance and the execution log were
downloaded and verified locally before any agent request.

- Total OA runner time: 28.13 seconds.
- Model loading: 18.65 seconds.
- Twelve-view forward inference: 1.58 seconds.
- Peak allocated GPU memory: 7,206,947,840 bytes, approximately 6.71 GiB.

Artifacts: `data/experiments/runs/appearance-agent-live-01/oa/` and
`oa-validation.json`. All cloud model calls use the user's previously confirmed
Gemini Free project, with no provider fallback.

## Initial attempts and request repair

The first agent request included 12 source views, 12 candidate renders and the
original photo. The model cited that photo, identified a two-hole paper punch,
and requested top views of candidates 1, 5 and 9. Blender executed the action.
The second model dispatch failed before a usable response was saved. The initial
trace's generic error category did not retain the precise provider error.

A separately recorded recovery attempt used only the two remaining dispatches
from the initial four-call budget. It again chose top renders of candidates 1, 5
and 9, then received **HTTP 400** on the following model request. Both original
jobs remain intact and have no completed canonical output.

Inspection found that every added view expanded string enums repeated inside the
nested identity and pose evidence arrays. Google's [structured-output documentation](https://ai.google.dev/gemini-api/docs/structured-output?hl=en)
notes that complex schemas, large enums and nested array constraints can produce
HTTP 400. This was a plausible diagnosis; the old adapter discarded the response
body, so the exact original 400 subtype is unconfirmed.

The repair moves the dynamic image-label inventory into the model context while
retaining small action/candidate enums and typed response fields. The harness still
rejects unknown evidence IDs, invalid tool arguments and unsupported poses before
execution/export. No permission, budget or semantic validation was relaxed.
HTTP error categories are now retained in traces, and known schema-complexity
messages are classified without storing arbitrary provider text or credentials.

A separate post-repair validation is limited to two new dispatches, for a maximum
of six across this development episode. This is a documented protocol change after
an implementation issue, not a replacement of the failed four-call record or an
independent accuracy comparison. OA predictions are reused from the fresh GPU run
above because the render bundle is unchanged.

Ruff, mypy and all **206 tests** passed after the repair. New tests verify that
32-image contexts do not grow nested label enums, unknown evidence is still rejected,
and HTTP 400 diagnostics do not expose provider text or credentials.

## Post-repair outcome

The first repaired request completed in 24.99 seconds. The model identified a
two-hole paper punch and requested top views of candidates **1, 6 and 10**.
Validation accepted the action and Blender produced all three images in 2.09
seconds. The following request included those images alongside the original
25-image evidence bundle, but failed with a transport error after 60.52 seconds.
No final pose decision or canonical mesh was exported. Semantic identification
was the model's hypothesis, not an independently verified annotation.

The recorded error is `transport`, not an observed HTTP 400. This does **not**
establish that the provider accepted the new schema or that the original schema
diagnosis was correct. The run stops at the declared combined six-dispatch limit;
there is no further cloud retry or provider fallback.

Artifacts under `data/experiments/runs/appearance-agent-live-01/`:

- `schema-fix-protocol.json`: revised budget and source versions.
- `schema-fixed-summary.json`: failed status, two dispatches, no final result.
- `schema-fixed-job/trace.md` and `events.jsonl`: requests and actual tool execution.
- `schema-fixed-job/work/evidence-1-1/renders/contact-sheet.jpg`: additional top
  views, left to right corresponding to candidate IDs 1, 6 and 10. The image's
  Pose 00/01/02 labels are local render indices, not candidate IDs or accepted poses.

This rerun demonstrates fresh OA inference on colored renderings, original-photo
delivery to the VLM, and a real agent-selected rendering action. It does not yet
demonstrate successful end-to-end canonicalization or an accuracy improvement.
The GPU inference process has finished; the rented server was not stopped.

## User-requested model switch

After the user requested another Gemini model, the live model-list endpoint was
queried and `gemini-3.6-flash` was selected. Google's pricing page listed its
standard input/output as free on the Free tier. This does not expose the project's
remaining quota; model listing alone does not establish generation entitlement.
The previously confirmed Free project was used, without changing billing or keys.

`model-switch-protocol.json` records a separate maximum of three new calls. The
same input bundle and fresh OA predictions were reused. Both actual generation
calls returned successfully with resolved model `gemini-3.6-flash`; no third call
was dispatched. The model first requested top views of candidates 4, 9 and 10,
then proposed candidate 4 with full-frame scope.

The final decision failed local validation: its cited evidence included
`candidate:4`, `source:1`, `detail:1:4:top` and `photo:capture-016`, but no contrasting
candidate or opposite view. The harness requires such a comparison. A local
replay of the saved response confirmed the rejection without another API call.
No canonical mesh was exported, and no evidence IDs were manually added.

This shows that this model could serve two substantial visual requests at that
time; it does not establish remaining daily quota for either model, resolve the
previous transport failure's cause, or establish semantic orientation accuracy.

Artifacts: `gemini-3.6-flash-job/trace.md`, `model-switch-summary.json`,
`model-switch-validation.json`, and `run-model-switch.py` under the run directory.
The repository model default and `.env.local` were not changed; this experiment
used the existing explicit model override. No production code changed in this step.
