# Orientation controls and support-plane results

## Outcome

The first support-surface experiment improves the hole punch's upright alignment.
The highest-ranked support plane has 10/12 up-direction votes within 25 degrees.
After applying a known arbitrary rotation to the input and rerunning OA, the
highest-ranked support plane maps back to the same up direction within numerical
precision. This is a two-orientation, single-object diagnostic, not a dataset-level
accuracy claim. Horizontal front direction remains unverified.

All 48 control images passed the numerical comparison between our runner and
the pinned official `inf_single_case` entry point. Known-pose chair results are
consistent with the coordinate convention, while still showing model prediction
error. These checks rule out an observed entry-point mismatch; they do not prove
that every integration assumption or every model prediction is correct.

## Controlled OA results

Code used for inference: `cce7bc5559ab850ef78fd757141f5bf614003858`.
Hardware: A100-SXM4-40GB; the existing locked OA environment and pinned weights
were reused. Each case has 12 rendered views and an additional official parity pass.
The known input rotation is Rx(41°) Ry(67°) Rz(-23°), using column vectors.

| Chair control | Median full rotation error | Median up-axis error |
| --- | --- | --- |
| Authored canonical pose | 17.32° | 14.52° |
| Known rotated pose | 7.04° | 5.48° |

The chair is procedurally authored test geometry: feet down, open seat toward +Z.
It is not an official evaluation sample. Reported errors use that authored frame;
they do not establish agreement with every learned category convention.

| Rotation-equivariance diagnostic | Median full-frame difference | Median up-direction difference |
| --- | --- | --- |
| Chair, original vs rotated | 20.09° | 17.96° |
| Hole punch, original vs rotated | 132.18° | 20.42° |

For each matching sampled camera index, compare R_rotated Q with R_base.
Changing the object pose changes which surface that camera sees. Full-frame
differences do not discount predicted yaw symmetry and must not be interpreted
as physical ground-truth rotation errors for the hole punch. Its front is unknown.

## Support-surface experiment

The original hole punch yielded nine stable-pose records; after near-duplicate
up-axis removal, eight distinct candidates were exported for review. Geometry
uses a uniform-density convex hull as a mass proxy. The top candidate has its
dark base below the light lever, visually matching the real video. An upside-down
candidate remains geometrically stable but receives zero OA up-direction votes.
Thus physical rest stability alone cannot choose the semantic upright pose.

| Diagnostic | Original input | Known rotated input |
| --- | --- | --- |
| Highest-ranked candidate's supporting up predictions | 10/12 | 10/12 |
| Median discrepancy from OA up predictions | 15.15° | 10.34° |
| Up-axis difference after undoing the known input rotation | <0.001° (paired comparison) | <0.001° (paired comparison) |

The small paired difference tests consistency of the selected support plane,
not its error relative to a measured real-world gravity vector. No such ground
truth has been measured. Thresholds remain provisional and the same OA predictions
are used for ranking; the vote count is not independent validation or a calibrated
correctness probability. The original generated mesh was already close to upright
in its native frame; this experiment mainly corrects the OA-induced tilts and
verifies recovery after deliberately rotating that mesh.

All original-input candidate exports preserve 316,954 vertices, 633,888 faces,
vertex colors and face indices. Recorded transforms reproduce the exported
vertices and place the lowest vertex at Y=0. No remeshing or regeneration occurred.

## Artifacts and next decision

Ignored local artifacts:

- `data/experiments/inputs/orientation-controls-01/`: fixture meshes, known rotations,
  rendered views and provenance.
- `data/experiments/runs/orientation-controls-01/`: raw predictions/logits, server logs
  and `evaluation.json`.
- `data/experiments/runs/support-video-5573-02/`: all support records, eight exported
  candidates, six-view previews, local geometry verification and `support-comparison.png`.
- `data/experiments/runs/support-punch-rotated-01/`: rotated-input support candidates
  and `equivariance.json`.

Use `support-video-5573-02`, which exports all eight distinct directions; the earlier
`support-video-5573-01` run exports only the first six. Both retain all raw planes.

The support module remains a diagnostic outside the production worker. Its output
is `needs_review`, `dataset_approval: pending_review`, `front_established: false`.
The next decision is to establish a category front convention and identify the
corresponding functional side from real imagery. The current displayed yaw is
only a projected OA representative; it can differ between inputs even when up
is recovered consistently. A broader set of real objects is required before
treating support-plane selection as a generally reliable upright estimator.

Local validation: 73 tests, Ruff and strict mypy passed. The implementation commit's
GitHub Actions run also passed. For method context and commands, see
[orientation diagnostics](../orientation-diagnostics.md).
