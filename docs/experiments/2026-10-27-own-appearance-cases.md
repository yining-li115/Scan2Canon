# New own-capture appearance cases

The requested cohort uses the current appearance-aware evidence agent, including
the model fallback list and bounded validation feedback. It is separate from the
earlier gray NOCS runs and the completed hole-punch example.

## Frozen selection

| Object | Video | Generation frame | Supplemental frame |
| --- | --- | --- | --- |
| Blue mug | IMG_5628.mov | frame-01.png | frame-00.png |
| Trigger spray bottle | IMG_5630.mov | frame-06.png | frame-01.png |

The mug exposes a handle and opening; the bottle exposes a trigger, nozzle and
label. Selection used the existing capture contact sheets before reconstruction
or orientation inference. Descriptive names are for experiment organization,
not category labels supplied to the open-semantic agent. Original captures remain
unchanged under `data/captures/2026-10-26-user-videos/`.

The input selection, frame/video hashes, manually selected segmentation boxes
and copied original frames are recorded in
`data/experiments/inputs/own-appearance-01/selection.json`. SAM 2 generates masks
from these boxes; masks must be inspected before SAM 3D generation. Generated
colors and unseen surfaces are model predictions. Supplemental frames are real
observations from the same video, not registered camera poses or scoring labels.

## Execution protocol

1. Generate and inspect the SAM 2 masks.
2. Run pinned SAM 3D on each primary image and mask with seed 42.
3. Download the resulting GLBs and inspect retained appearance.
4. Render twelve known-camera views with appearance admission enabled.
5. Run fresh OA predictions for those exact render bundles on the GPU server.
6. Bind original and supplemental photos to each generated mesh hash.
7. Run the current evidence agent locally, initially with Gemini 3.6 Flash and
   the configured fallback pool, maximum four calls per object/eight in total.
8. Preserve every failed attempt. Inspect exported meshes only if validation passes.

This is a development diagnostic, not an independently annotated accuracy test.
No canonical angle is assumed from the generation model or source-camera pose.
No gray-mesh substitution or invented texture is permitted. A successful run
does not approve the object for dataset admission.

The local runner is preserved at
`data/experiments/runs/own-appearance-01/run-canonicalization.py`. It refuses to
start before reconstruction, OA and observation manifests exist. It does not
automatically retry transport failures or overwrite an existing job.

## Prerequisites observed

The A100 40 GB server was reachable. Its OA environment was already available,
but the SAM 3D environment and weights were absent. The repository's pinned
setup script restored its isolated environment; host GPU drivers
and the existing OA environment are unchanged. The local and remote setup-script
and SAM 3D lockfile hashes matched before installation.

The pinned SAM 3D checkpoint access check returned HTTP 401. The new server did
not have the previous Hugging Face login in its standard checked cache locations.
The user completed login with the already-approved account. The pinned checkpoint
then downloaded successfully and its file hashes were recorded. The isolated
environment passed the official imports and CUDA kernel check. Both seed-42
reconstruction runs have been started sequentially on the A100.

No new canonical output is claimed until the corresponding stage manifests and
final validated export exist.

## Completed input stages

SAM 2.1 small segmentation completed for both objects with the pinned source and
checkpoint. Prepared RGB/mask inputs, mask overlays and provenance were downloaded
locally under each input's `segmentation/` directory. Binary-mask, dimension,
orientation and input-checksum checks passed. Visual review found the main object
silhouettes retained; the mug handle-hole boundary remains soft and should be
checked in the generated mesh. This is a qualitative mask review, not measured IoU.

The review is saved in `data/experiments/runs/own-appearance-01/mask-review.json`.
The SAM 2 source used for this early preparation was independently checked out at
the same pinned revision in `data/tools/sam2-segmentation-01` while SAM 3D extension
compilation continued.

## Reconstruction and orientation evidence

Both SAM 3D runs succeeded. All exported reconstruction assets and hashes are
preserved locally, including the colored mesh, Gaussians, raw mesh and predicted
layout. The GLBs store vertex colors, not baked UV textures.

| Case | SAM inference | Peak allocated GPU memory | Mesh vertices / faces |
| --- | ---: | ---: | ---: |
| Mug | 19.70 s | 17.89 GiB | 344,926 / 689,836 |
| Spray bottle | 14.15 s | 17.38 GiB | 186,356 / 372,696 |

These times exclude setup, checkpoint verification and model loading. Qualitative
render review found the mug opening and handle hole retained, with no readable
logo on the generated mesh. The spray bottle retained label colors, nozzle and
trigger structure. Neither review measures reconstruction accuracy.

Each mesh passed appearance admission, received twelve fresh colored renders and
fresh OA inference. The local loader verified the exact render manifest, prediction
coverage, logits and checkpoint provenance. The server used its pinned existing OA
runner; the local current renderer and appearance admission supplied the evidence.
OA symmetry outputs disagreed across views: mug fits were 0/2/4 and spray fits were
0/1/2/4. These are model hypotheses, not measured object symmetry.

Each observation manifest binds two original frames to the generated mesh hash.
The agent trace confirms both original frames were attached on its first turn.
Full stage metrics are in `data/experiments/runs/own-appearance-01/stage-metrics.json`.

## Live agent results

Both cases completed with Gemini 3.6 Flash, three calls per case (six total), with
no request for user confirmation. The Free-tier project remained unchanged. No
429 occurred, so the configured fallback pool was available but not exercised.
These two selected cases do not establish an automatic acceptance rate.

| Case | Executed evidence actions | Result | Qualitative limitation |
| --- | --- | --- | --- |
| Mug | `lookup_convention(mug)` | Full-frame proposal, candidate 10; handle toward +Z | Visible residual body/base tilt remains |
| Spray bottle | `lookup_convention(bottle)`, `render_view(top, [1, 2, 8])` | Full-frame proposal, candidate 1; main label toward +Z | Label front is an instance proposal; nozzle front remains a different meaningful convention |

The mug's second reply populated `candidate_ids` on a `finish` action, which
violated the action contract. The harness rejected it without executing an export,
returned validation feedback and accepted the corrected third reply. This is a
real bounded correction run, not a scripted demonstration. The geometry and pose
were not manually corrected after seeing the result.

The spray bottle agent explicitly distinguished the nozzle-facing candidate from
the label-facing candidate after requesting top views. The retrieved generic
bottle convention defines up and leaves front unspecified. The agent's final
label-front meaning is therefore its own instance proposal, not a newly approved
spray-bottle category convention. The example reveals the outstanding need to
evaluate and govern consistent front meanings across instances. The agent's
claim that no evidence was missing does not certify that consistency.

Both rigid exports passed geometry and appearance preservation checks. Maximum
inverse-transform vertex errors were 1.28e-8 (mug) and 1.26e-8 (spray bottle) in
mesh units; these measure export fidelity, **not semantic orientation error**.
Scale remains unverified and unchanged. Both outputs retain
`dataset_approved: false`. There are no independent pose labels for these cases,
so no angular accuracy or SOTA comparison is reported.

Artifacts under `data/experiments/runs/own-appearance-01/`:

- `summary.json`: call counts, executed actions, rejected reply and final status.
- Each case's `job/trace.md` and `events.jsonl`: readable and machine-readable logs.
- Each case's `job/work/agent-export-3-1/`: canonical GLB and transformation report.
- Each case's `canonical-preview/contact-sheet.jpg`: six labeled inspection views.
- `server-logs/`: setup, download, segmentation and model inference logs.
- `visual-review.json`: post-run visual inspection, kept separate from inference.

Local verification: Ruff passed; mypy passed across 108 source files; 81 relevant
tests passed for reconstruction, appearance, open canonicalization, the evidence
agent and Gemini transport. The experiment scripts also passed Ruff. No production
source was changed for these runs, and no commit was created.
