# OA input-condition diagnostic

This follow-up asks whether the two mugs' inconsistent frames come primarily from
the supplied images, the OA forward/decoder path, or our coordinate conversion.
It preserves the [fresh GPU trial](2026-10-29-fresh-gpu-pipeline.md) and changes no
canonicalization decision, threshold or category definition.

## Protocol

The first batch contains 59 images: six preserved real photographs (two blue,
four red), two foreground crops, 24 unchanged spherical renders, 24 new ring
renders and three images distributed with the pinned upstream repository.
All prepared views are retained, including handle occlusion. Ring cameras use
an elevation of `atan(0.35)`, approximately 19.3 degrees, with twelve yaw positions.
The meshes, colors, lighting and render settings are otherwise unchanged.

Foreground crops use the existing reconstruction masks, their foreground bounding
boxes and square padding with an 85% long-side occupancy. They do not invoke the
demo UI's optional rembg preprocessing. Raw scene photographs contain distracting
background objects, so their failure alone is not evidence of single-object model
failure. Photographs are unregistered and never converted to executable mesh poses.

Visual inspection found that the red reconstruction mask removes some interior
coffee pixels. A separate two-image follow-up uses the same bounding boxes and
padding but retains every original RGB pixel. This follow-up was added after
observing the masking defect; it is development, not a predefined holdout test.

`experiments.orientation.image_probe` validates image hashes and invokes the same
forward function as registered mesh inference. Its separate manifest and output
formats cannot be passed to the mesh solver as registered views. Every image is
also checked against upstream `inf_single_case`; all 61 pass. Failures and partial
outputs are retained, and an existing probe output cannot be overwritten.

## Observations

Counts below are official decoded fits, not accuracy measurements. Fit 0 means
the decoder did not accept a 1/2/4-period fit; it is not a verified symmetry label.

| Input | Blue: fits 0 / 1 / 2 / 4 | Red: fits 0 / 1 / 2 / 4 |
| --- | --- | --- |
| Raw photographs | 2 / 0 / 0 / 0 | 4 / 0 / 0 / 0 |
| Masked crop | 0 / 0 / 1 / 0 | 1 / 0 / 0 / 0 |
| Bounding-box crop, no masking | 1 / 0 / 0 / 0 | 1 / 0 / 0 / 0 |
| Spherical renders | 8 / 0 / 3 / 1 | 7 / 2 / 2 / 1 |
| Near-level ring renders | 7 / 0 / 1 / 4 | 10 / 1 / 1 / 0 |

The official F35 example produces fit 1; the hat and bottle examples produce fit
0 under the tested preprocessing. These examples are execution controls without
audited reference directions, not a measured demonstration success rate.

For known-camera renders, pairwise differences between recovered up directions
were measured in the same original mesh coordinates, using all twelve views:

| Object | Sphere median / maximum | Ring median / maximum |
| --- | --- | --- |
| Blue | 23.92 / 133.09 degrees | 22.50 / 31.83 degrees |
| Red | 25.12 / 110.74 degrees | 22.71 / 32.11 degrees |

Near-level views reduce the largest disagreements but leave substantial median
disagreement. These are cross-view consistency measurements, not ground-truth
pose errors. Neither cropping nor changing camera elevation solved full-frame
consistency for these objects.

### A concrete decoding loss

Red ring view 3 has a clearly visible handle and an azimuth-score peak at 83
degrees. Its maximum sigmoid score is 0.6558. The best official periodic fit has
concentration 0.6256, above the single-direction threshold of 0.6, but R-squared
0.44857, just below the required 0.45. The decoder therefore returns fit 0.

This shows that fit 0 can discard a directional signal; it does not establish
that the peak is semantically correct or that thresholds should be lowered.
Other failures have genuinely diffuse or incompatible fitted distributions.
All raw logits and fitted-mode diagnostics are preserved. Scores are not
calibrated probabilities of correctness.

## Implementation checks and limits

- All 24 unchanged spherical images reproduce the previous run's decoded outputs
  and bitwise-identical logits after the forward-function extraction.
- The official Blender axis asset passes the camera, axis basis, Euler order and
  inverse-pose calibration again. This reduces concern about those coordinate
  equations but does not prove every pipeline operation correct.
- No VLM/agent was called and no existing canonical result was changed.
- No reference pose or manually chosen front was fed into inference.
- The current per-image decoding and modal-fit aggregation lose distribution
  information. A useful next experiment is to combine complete directional
  evidence across calibrated views and preserve competing modes, then compare
  against this baseline. Its success has not been demonstrated here.

## Artifacts and reproduction

Local root: `data/experiments/runs/oa-input-diagnostic-01/`.

- `protocol.json`, `prepare_probe.py`, `inputs/images.json`: frozen first-batch inputs.
- `oa/`: all 59 predictions, raw logits, official parity and checkpoint provenance.
- `box-control-inputs/`, `box-control-oa/`: separate two-image masking control.
- `box-control-axes.png`, `box-control-visualization.json`: both follow-up predictions
  with axis insets, raw score curves and checksums.
- `analysis.json`, `analyze_probe.py`: all fits and known-camera consistency metrics.
- `input-comparison-axes.png`: representative photos, crops and renders with axis
  insets and directional-score curves; all individual plots are in `plots/`.
- `plots/{blue,red}-ring-axes.png`: every ring view with predicted axes.
- `forward-refactor-parity.json`, `coordinate-calibration.json`: implementation controls.

Axis insets use an orthographic camera-space glyph: +Y up in green, +Z front in
blue and +X in red. They do not locate the object's origin in the photo. For fit
0/2/4, dashed horizontal axes show an unresolved or multiple-direction branch.
They are predictions, not independently validated canonical frames.

To run a prepared diagnostic bundle on the configured server:

```bash
experiments/orientation/.venv/bin/python -m experiments.orientation.image_probe \
  data/experiments/runs/oa-input-diagnostic-01/inputs \
  /path/to/new-probe-output
```

Plots used a temporary Matplotlib environment; no project or GPU dependency was
added. Ruff, formatting and mypy passed; all 271 tests passed, including tampered
image rejection before model setup, duplicate IDs and interrupted-probe preservation.
All images and result arrays are available locally; the GPU server remains running.
