# Rotation consistency: 27 October 2026

The interactive visual-agent workflow recovered the same up direction for the
bottle, bowl, can and mug across five input rotations each. The mug also recovered
the same complete orientation. The camera and laptop remained unresolved in all
five rotations. Abstentions are not successful orientation recovery.

## Protocol

- Six previously selected NOCS holdout objects, five prepared rotations each.
- The first rotation reuses its saved result from the earlier holdout experiment.
  The other 24 decisions are new. No decision was replaced after scoring.
- Selection used the frozen skill, conventions, geometry tools and renderer from
  commit `6dbe053f4cbab460fc25d0900ecb782b7dcf2c7f`. Their hashes, all case IDs,
  input hashes, reused-result hashes and the 15° threshold were saved before
  inspecting new candidates.
- The interactive agent inspected candidate images, submitted validated up
  decisions, and inspected front candidates for the four new mug cases. No
  per-object human confirmation or new GPU inference was used.
- Every decision was saved before reading the known rotations and scoring
  references. The same agent retained prior conversation context and saw related
  instances together: this is not an independent blinded model evaluation.

The selection pipeline was unchanged during this experiment. These are the
interactive agent's results, not results of OA V2 or the local Qwen model.

## Measurement and results

Let `Q` be the known source-to-input rotation and `R` the predicted
input-to-canonical rotation. Each recovered frame in the original object's
coordinates is `C = R @ Q`. Comparing the raw `R` matrices would be incorrect
because each input uses a different coordinate frame.

For each object, compare all ten pairs of recovered frames. For an up-only
convention, compare the +Y directions; yaw remains unspecified. For a full frame,
also compare complete rotations using their geodesic angle. Report errors against
the dataset reference separately: a consistently wrong decision can still be
rotationally consistent.

| Object | Proposals / rotations | Compared axes | Maximum pairwise difference | Maximum reference error |
| --- | ---: | --- | ---: | ---: |
| Bottle | 5/5 | Up | <0.001° | 1.15° up |
| Bowl | 5/5 | Up | <0.001° | 0.30° up |
| Can | 5/5 | Up | <0.001° | 2.67° up |
| Mug | 5/5 | Full frame | <0.001° | 0.29° up; 6.18° full frame |
| Camera | 0/5 | Unresolved | Not measurable | Not measurable |
| Laptop | 0/5 | Unresolved | Not measurable | Not measurable |

Overall: 20/30 cases produced a proposal, comprising 15 up-only and five full-frame
exports. Ten cases abstained. All 40 measurable up pairs and all ten measurable
full-frame pairs met the predeclared 15° threshold. Coverage is 4/6 objects, with
only one object testing complete orientation. Repeated rotations are correlated
observations, not 30 independent objects.

The near-zero pairwise differences reflect choosing corresponding geometric
candidates on exactly the same mesh. They do not imply near-zero semantic error.
The mug's approximately 6.18° reference offset remains in all five outputs.
The untextured camera lacks clear top/base features; the laptop lacks reliable
keyboard-versus-display evidence. Neither gained evidence merely by being rotated.

## Scope and limitations

This validates rotational consistency, not translation/origin equivariance,
reconstruction quality, repeated independent scans, unknown categories, or
correctness on OO3D-9D and user captures. Output previews share a viewing direction
but are independently framed; apparent image size can differ despite unchanged
mesh scale. The exporter derives its horizontal origin from the input bounding
box, so identical output positions are not established. Scoring references are
audited dataset coordinates, not independent human semantic annotations. Dataset
admission remains ungranted.

The next useful test is evidence enrichment for unresolved objects, followed by
different objects and real captures. Repeating rotations of the same easy object
does not measure that generalization.

## Reproduction and artifacts

Raw records remain under the ignored local directory
`data/experiments/runs/nocs-rotation-consistency-01/`:

- `protocol.json`: frozen cases, versions and previously saved result hashes.
- `case-*-up/`, `case-*-front/`: candidates and rendered evidence.
- `case-*-up.json`, `case-*-front.json`: actual visual decisions and observations.
- `case-*/`: validated results and canonical GLBs where resolved.
- `evaluation.json`: final scored result; the earlier `consistency.json` remains
  preserved from before adding the frozen-file verification guard.
- `previews/`, `comparison.jpg`, `mug-comparison.jpg`: renders and montages.
- `comparison-manifest.json`: montage source-image and output hashes.

The tracked [results JSON](2026-10-27-rotation-consistency-results.json) includes
the frozen protocol, individual result hashes, common-frame matrices, pairwise
angles and separate reference errors. The first rotation's raw artifacts remain
in `nocs-agent-holdout-01/` and are linked by the protocol.

Re-evaluate saved decisions into a new file:

```bash
uv run --locked --extra canonical python -m experiments.canonicalization.consistency \
  data/experiments/runs/nocs-rotation-consistency-01/protocol.json \
  data/datasets/scan2canon-mini-v1/nocs-rotations-02 \
  data/experiments/runs/nocs-rotation-consistency-01/rechecked.json
```

The evaluator checks all saved decisions before opening answer-bearing references,
rejects changed frozen tools and mismatched meshes, verifies exported artifact
hashes and proper rotations, and refuses to overwrite a previous report.
Regression tests cover transform composition, unspecified yaw, abstentions,
consistent-but-wrong poses and missing decisions before reference access.
