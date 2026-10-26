# First OA V2 run: generated hole-punch mesh

## Outcome

The existing SAM 3D mesh completed the render → OA V2 → pose candidates → GLB
export pipeline. The semantic result is **needs_evidence**, with dataset approval
**pending_review**. This is an executed integration experiment, not an accepted
canonical orientation or a model accuracy benchmark.

The 12 predictions split evenly between symmetry fits 2 and 4. Neither fit is
confirmed physical symmetry. The solver retains both tied interpretations rather
than choosing whichever appears first in the input. Candidates are ranked within
each fit group; candidate numbering does not rank the groups against each other.

| Candidate | Fit | Source view | Supporting views | Support out of 12 |
| --- | --- | --- | --- | --- |
| 01 | 2 | 04 | 01, 02, 03, 04 | 4 |
| 02 | 2 | 07 | 01, 07 | 2 |
| 03 | 2 | 00 | 00 | 1 |
| 04 | 4 | 08 | 05, 06, 08, 09, 10 | 5 |
| 05 | 4 | 11 | 11 | 1 |

Support uses a 25-degree rotation distance and each group's predicted yaw
equivalence. It is not exact-pose agreement or a correctness probability. Visual
inspection of the first fit group shows residual tilt and an incorrect upright
hypothesis. No automatic acceptance is justified.

## Execution and provenance

- GPU: NVIDIA A100-SXM4-40GB, driver 590.48.01.
- Runtime: Python 3.11.17, PyTorch 2.7.1+cu128, BF16 autocast.
- Inference project commit: `b64ad4c384a179f833064c9a955ade691863bbd9`.
- Upstream source: `73b11c9dc83e84daeb563d0c766831f2c66b0a18`.
- Checkpoint: `Viglong/OriAnyV2_ckpt`, revision
  `34fe2ecbdb987a0fb98437af00600f22a0fdfb99`.
- Input: `data/experiments/inputs/video-5573-orientation-02/`.
- Raw inference: `data/experiments/runs/orientation-video-5573-01/`.
- Final candidate export: `data/experiments/runs/canonical-video-5573-02/`.
- Per-candidate previews: `candidate-01-preview/` through `candidate-05-preview/`
  inside the final export directory.

The earlier `canonical-video-5573-01/` export is retained as a diagnostic of the
tie-selection issue. Use `-02` for review. Its report records the corrected solver's
SHA-256; reusing the same raw predictions does not require another GPU pass.

| Measurement | Observed value |
| --- | --- |
| Model construction, checkpoint loading and GPU transfer | 15.30 seconds |
| All 12 sequential view predictions, including decoding and output writes | 1.57 seconds |
| Complete inference command's recorded run | 25.04 seconds |
| Peak PyTorch allocated GPU memory | 7,206,947,840 bytes (6.71 GiB) |

This is one run at 518 × 518, batch size one, not a steady-state throughput
benchmark. Environment installation, checkpoint download, rendering, network
transfer and candidate export are excluded. Allocated memory excludes the CUDA
context and other process overhead; it is not a guaranteed hardware minimum.

## Validation and remaining evidence

The local verifier checked the input/render hashes, raw logits hash and all five
exported meshes. Each retains 316,954 vertices, 633,888 faces and identical vertex
colors and face indices. Transformed vertices match the recorded rigid transform
within 1.4e-8 model units. See `local-verification.json` for artifact hashes.

The tie regression test verifies that reversing input prediction order produces
the same result and that both fit groups retain their own supporting views.
The full local suite passed 71 tests, Ruff and strict mypy.

Before accepting a canonical mesh, establish the hole-punch category's front/up
convention, confirm the paper-insertion side from real captures or contributor
input, and validate upright alignment. Generated bottom views are not new physical
observations. A future agent can request that evidence; this experiment currently
records the unresolved decision but does not implement an LLM/VLM agent or a
contributor confirmation interface.
