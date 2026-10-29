# Cross-instance mug consistency: IMG_5654

This development run tests whether a newly captured white patterned mug with a red
handle is independently identified and oriented consistently with the earlier blue
mug. The current evidence-agent code is frozen for this run; no target class or
previous mesh pose is supplied to the model. Existing pilot conventions remain
available through `lookup_convention`.

## Frozen protocol

- Preserve the 9.40-second original video under
  `data/captures/2026-10-27-mug-5654/` with its checksum and ffprobe metadata.
- Extract nine upright PNG frames using ffmpeg autorotation and `fps=1`.
- Select frame 01 before reconstruction for its visible rim, body and handle.
  Preserve frames 00, 03 and 08 as additional original observations. Some additional
  frames crop the object. The underside is not directly observed; liquid fills
  part of the opening.
- Use the existing pinned SAM 2 box segmentation, inspect the mask, then run
  pinned SAM 3D with seed 42. No new texture or geometry is hand-authored.
- Render twelve colored mesh views and run fresh OA predictions on those exact
  views. Bind the four photographs to the generated mesh hash.
- Run the existing Gemini evidence agent, maximum four total Free-tier provider
  calls including fallback and validation corrections. Preserve failures and raw
  replies. No pose correction is applied after inspecting the result.
- After inference, compare function, declared front meaning and actual mesh frame
  against the project mug convention: +Y from base to rim, +Z toward the handle.
  The comparison is qualitative, with no independently annotated pose ground truth.

Inputs: `data/experiments/inputs/mug-consistency-5654-01/`.
Outputs: `data/experiments/runs/mug-consistency-5654-01/`.
Previous reference: `data/experiments/runs/own-appearance-01/mug/` and its separately
recorded up-refinement diagnostic. Cross-instance views must distinguish that
previous offline diagnostic from a fresh autonomous run.

## Completed results

The mask retains the rim, body and red handle. Most liquid is excluded, but some
foam fragments are included; this was recorded before reconstruction without
manual mask edits. SAM 3D produced a colored open mug with a handle hole and
generated bottom. The liquid was not represented as a separate surface. Cavity
depth, underside and fine text remain unverified.

SAM 3D inference took 17.56 seconds after 41.06 seconds of model loading on the
A100 40 GB. The export contains 349,776 vertices and 699,300 faces, with vertex
colors rather than UV image textures. All reconstruction artifact hashes were
verified after downloading. Fresh OA inference succeeded for twelve colored views
and passed official single-case entry-point parity.

The fresh Gemini 3.8 Flash run completed in three calls with no fallback, rejected
reply, requested user confirmation or manually selected pose:

1. Independently identify a ceramic coffee mug and `lookup_convention(mug)`.
2. `refine_up([1, 2, 6])`, inspecting three support-refined alternatives.
3. Select candidate 14, derived from parent 6, and export a full-frame proposal.

The selected up correction was 17.10 degrees. Near-contact polygon coverage rose
from 0.80% to 61.98%; these are mesh support diagnostics, not semantic ground-truth
accuracy. The rigid export preserves shape and stored appearance. The inverse
vertex error was approximately 1.9e-16 mesh units; scale remains unverified.

The final functional identity is **ceramic coffee mug with handle**. Its declared
front is horizontally from the body toward the red handle, explicitly referencing
the retrieved mug convention. Post-inference front/top render inspection agrees:
the opening is upward and the red handle lies approximately along +Z. The previous
blue mug's actual refined frame uses the same broad directions, although its
handle retains a visible yaw offset and its previous model description was
inconsistent. That earlier textual failure has not been erased or corrected here.

Thus this example supports cross-instance **functional category and directional
convention agreement**. It does not establish exact angular agreement or a general
success rate. The previous blue result is a saved-reply diagnostic using Gemini
3.6 Flash; the new result is a fresh Gemini 3.8 Flash run. This is not a controlled
same-model benchmark. Neither output is approved for dataset admission.

## Inspect locally

Under `data/experiments/runs/mug-consistency-5654-01/`:

- `two-mugs-canonical.png`: blue mug on the left, new mug on the right; front above,
  top below, with +X red, +Y green and +Z blue. Meshes are independently framed,
  so apparent dimensions must not be compared as metric sizes.
- `job/work/agent-export-3-1/canonical.glb` and `canonicalization.json`: unchanged
  selected result, rigid transform, semantic descriptions and appearance checks.
- `job/trace.md`, `events.jsonl` and `calls/`: three actual requests and tool history.
- `comparison-provenance.json` and `visual-review.json`: post-inference comparison,
  artifact hashes and limitations, separate from the model's decisions.
- `local-code-snapshot.json`: inference code and convention hashes, verified
  unchanged throughout the run. No production code was modified for this example.

The original video, selected images, segmentation, generated assets, OA predictions,
logs and canonicalization artifacts are preserved locally. GPU jobs have completed;
the server has not been stopped. Ruff check/format, mypy and all 227 tests passed.
