# OA fused top-1 stability and input-order repair

The two development mugs retain the same broad handle side across the examined
top-1 fits. The result is not a measurement of canonical pose accuracy: the
largest complementary-half frame difference is 25.71 degrees, and independent
part/frame labels are unavailable. No acceptance threshold or agent export gate
was changed.

## Method

This evaluates **our marginal fusion of saved OA V2 outputs**, not an official
single-image OA accuracy metric. Each mug has twelve spherical renders and twelve
near-horizontal ring renders of the same colored mesh. The original photos are
not registered or numerically fused. There is no new GPU inference, reconstruction,
cloud-model request or support-plane refinement in this experiment.

For each of the four object/camera conditions, the protocol fixes:

- One all-view fit.
- Six complementary half splits, six views per half: the existing even/odd split
  and five unique random splits with seed 20261030.
- Twelve leave-one-out fits, each compared with the all-view result.

This gives 100 fits, with no reference labels used to select candidates. We compare
the top-ranked rotation, its up vector and its front vector without symmetry
reduction. Missing fits are reported explicitly. The half-pair members are disjoint,
but different trials share views; leave-one-out overlaps the all-view result.
These are descriptive statistics, not independent samples or confidence intervals.

The official decoded per-view matrices and symmetry fits remain in each summary
as a separate diagnostic. A fit-zero decoded result does not establish yaw.
The matrices from different objects are never directly compared as semantic errors.

## Input-order defect and fix

The first run is preserved in `data/experiments/runs/oa-top1-stability-01/`.
An additional post-hoc control reversed the input order or rotated it by six
positions, keeping all image logits and camera pairings identical. The v1 search
changed its first camera and thus its finite set of initialization poses. Full-frame
differences reached 9.56 degrees. This was solver sensitivity, not new OA evidence.

`oa-marginal-fusion-v2` orders evidence by a SHA-256 digest of its float64 logit
content, with camera bytes breaking identical-logit ties, before seed generation
and numerical search. There is no semantic or category preference in this ordering.
Per-view scores are mapped back to the caller's original order. Identical rows
remain interchangeable. Input ordering invariance does not make the finite optimizer
globally optimal or certify pose accuracy.

The entire fixed protocol was rerun in a new directory,
`data/experiments/runs/oa-top1-stability-02/`, and both order controls were repeated
for all four conditions. All eight reordered top-1 comparisons are now identical
(reported frame difference 0.0 degrees). Some subset differences became larger:
the fix removes order dependence, not all search or observation uncertainty.

## Final v2 results

All angles below measure prediction-to-prediction differences, **not GT error**.

| Object / cameras | Half-pair frame median / max | Half-pair front max | Leave-one-out frame median / max |
| --- | ---: | ---: | ---: |
| Blue / sphere | 10.96° / 12.27° | 11.99° | 3.84° / 10.71° |
| Blue / ring | 8.20° / 10.38° | 6.73° | 3.66° / 10.20° |
| Red / sphere | 12.63° / 16.35° | 16.34° | 4.49° / 11.18° |
| Red / ring | 12.15° / 25.71° | 25.51° | 4.04° / 9.79° |

All 100 fits returned a top candidate. The largest subset-to-all frame difference
was 22.66 degrees. None of the examined top-1 fits flipped to the opposite frame.
Comparing all twelve spherical views with all twelve ring views of the same mesh:

| Object | Full-frame difference | Up difference | Front difference |
| --- | ---: | ---: | ---: |
| Blue | 3.68° | 3.06° | 2.61° |
| Red | 5.75° | 5.52° | 1.67° |

The diagnostic threshold sweep was specified before the first run. Of the 24
complementary-half pairs, 0 fall within 5 degrees, 9 within 10, 22 within 15 and
23 within 20. These counts describe sensitivity; they are not semantic precision,
accepted-object coverage or evidence for choosing a deployment threshold.

## Cross-instance visualization

The all-view figure shows each mug under both camera conditions, from the proposed
front and from above, with red X, green Y and blue Z axes. Both mugs place their
handle broadly opposite positive Z. The worst-split figure includes each mug's
largest half-pair disagreement, selected only after all fits were saved. It retains
the same broad handle side while exposing tilt and yaw drift.

This qualitative review was performed by the interactive visual agent, not an
independent human annotator or the automated evidence agent. Its observations,
image hashes and limitations are recorded in `visual-review.json`. No independently
annotated 3D landmarks establish a cross-instance angular error. Dashed axes denote
unverified proposals; these are raw OA-fusion poses, not accepted canonical exports.

## Implications and boundaries

On these objects, the presence of a lower-ranked opposite candidate was not enough
to show that the leading branch was unstable. A future agent decision should
consider support and stability of competing branches, rather than treating every
retained search candidate as equally plausible. This experiment does not yet change
that policy or replace the prior up-only results.

The remaining angular drift matters even without a 180-degree flip. Stability can
also be consistently wrong. Calibrating acceptance requires held-out objects with
independent frame/part annotations, measuring error and abstention together. The
two familiar mugs, correlated renders and post-hoc solver repair cannot supply
that calibration. Changing reconstruction, original capture viewpoint, mesh quality
or category distribution remains untested here.

## Reproduction and verification

```bash
uv run --locked --extra canonical python -m experiments.orientation.stability \
  /path/to/verified/views /path/to/verified/oa /path/to/new-diagnostic
```

Use `--resume` only with unchanged inputs, protocol and implementation. Completed
fits are checksummed and reused; a failed fit has one further attempt. Changed
artifacts are rejected. Each job retains its request, stage events, all candidate
fits and summary. Experiment scripts, source snapshots, control protocols and
axis figures remain in the local run directories.

Ruff, Ruff format, mypy (`src tests experiments`) and 294 tests pass. New tests cover
front flips versus up agreement, disjoint split construction, preserved missing
fits, interrupted-stage recovery, artifact corruption, input reordering invariance
and per-view score correspondence. The tests use synthetic evidence and do not
claim model accuracy.
