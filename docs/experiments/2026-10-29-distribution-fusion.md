# OA marginal fusion and bounded agent review

This development experiment adds a CPU candidate generator that scores one
mesh-space rotation against every registered view's saved OA distributions. It
improves some split-view consistency measurements but does not establish a shared
semantic front for the two mugs. The decoded proposal route remains the default.

## Method and boundaries

For each registered render, OA supplies 360 azimuth, 180 elevation and 360 roll
logits. We apply sigmoid and normalize each axis's bins, then interpolate their
log scores at the angles implied by a candidate rotation and the saved camera.
The score is the equal-axis, equal-view mean log score relative to a uniform bin
score. It is a ranking heuristic: Euler marginals are factorized, generated views
are correlated and scores are not a calibrated joint posterior or confidence.
No `symmetry_fit` label removes a view from this calculation.

Search uses 4,096 seeded global rotations relative to the first camera, plus
per-view argmax poses at twelve yaw offsets. Eighteen separated seeds undergo
bounded Powell refinement. Up to six separated proposals are retained, with
per-view scores, score gaps and optimizer status. Search is finite and does not
certify that all modes were found. The 25-degree seed separation and 15-degree
candidate separation are search settings, not correctness thresholds. Flat
marginals return no proposals. Euler poles marginalize the coupled azimuth/roll
family; periodic bins wrap, and elevation uses the available -90 through +89 bins.

There are no category rules, manually supplied upright axes or reference poses.
The optimizer makes continuous rigid rotation proposals; it does not alter mesh
geometry, texture, OA weights or official fit thresholds. `front_available`
means a candidate direction exists for inspection, not that it is semantically
established. The agent must still pass the existing evidence and direction checks.

## Inputs and protocol

All inputs come from the preserved [61-image diagnostic](2026-10-29-oa-input-diagnostic.md).
There is no new GPU inference. For each mug, both the original spherical views
and near-level ring views are included. The original photos remain unregistered
visual evidence; they never enter the numeric mesh-space fusion.

`experiments.orientation.bind_probe` extracts only render predictions whose mesh,
source manifest, image hash, view ID, camera orientation and position exactly
match a registered bundle. It preserves logits without recomputation, records
probe-to-view IDs and refuses photos without that provenance. An existing output
cannot be overwritten and failed bindings retain their status.

Before scoring, `protocol.json` fixed even-indexed views as subset A and odd-indexed
views as subset B. Each subset has six views and is fitted independently. All
twelve views are also fitted for the agent proposal pool. No candidate was selected
using reference directions or these comparison metrics. These are two familiar
development objects, not a held-out benchmark.

## Split-view results

The baseline is the top decoded hypothesis from `orientation.solve.hypotheses`,
computed separately for each subset. It is not the prior agent's final refined pose.
The first retained baseline candidate has fit zero in every subset, so its
horizontal axes remain unresolved. Red/sphere subset A has tied fit modes 0/1/2;
the baseline retains all tied modes, and this comparison uses its first candidate.
A baseline full-frame accuracy comparison would be misleading.

| Object / cameras | Baseline up difference | Fused up difference | Fused full-frame difference |
| --- | ---: | ---: | ---: |
| Blue / sphere | 32.25° | 3.17° | 4.37° |
| Blue / ring | 29.66° | 7.77° | 10.11° |
| Red / sphere | 19.79° | 8.21° | 16.73° |
| Red / ring | 1.03° | 9.85° | 11.42° |

These numbers compare two independently fitted view subsets, not predictions
against ground truth. Fusion improves up consistency in three conditions and
worsens it in the fourth. Its full-frame differences do not certify semantic
front correctness, and the two objects' input coordinate matrices cannot be
compared directly as a cross-instance semantic error.

## Actual agent execution

Both mugs use their original spherical views, original-photo observation bundles
and the fused six-candidate pool. Each job has a four-call budget; each finishes
in two Claude calls. No model switch, extra retry or manual pose replacement is
used. In both jobs the agent calls `refine_up`, inspects the resulting images and
finishes with candidate 6 as an **up-only** proposal:

| Object | Parent candidate | Up correction | Final scope |
| --- | ---: | ---: | --- |
| Blue | 0 | 4.86° | `up_only` |
| Red | 0 | 6.39° | `up_only` |

Correction is measured relative to the parent, not relative to ground truth.
Both agents describe base-to-rim as up and retain handle-side versus graphic-face
front hypotheses. Neither selects a front, requests a capture or creates a shared
category definition. Their projected up checks pass only 2D compatibility using
points estimated by the same model; these are not independent annotations.
The raw blue response also contains approximate angle statements about the fronts
and candidates that are not established measurements. They are preserved as model
claims rather than promoted to verified geometry.

The final previews use solid green Y and dashed red X / blue Z. Apparent horizontal
alignment in those pictures is a representative yaw, not proof that a common
semantic front has been chosen. Both rigid exports preserve the input geometry
and appearance under the existing export checks.

## Running the optional route

Use the existing verified render and OA prediction directories:

```bash
uv run --locked --extra canonical --env-file .env.local \
  python -m experiments.canonicalization.agent_worker \
  /path/to/views /path/to/oa /path/to/new-job \
  --backend claude --proposal-method distribution --max-calls 4 \
  --observations /path/to/observations --blender /path/to/blender
```

`--proposal-method decoded` remains the default. Proposal method and fusion source
version participate in resume validation. Switching methods requires a fresh job.
A flat or insufficient pool stops before candidate rendering or a model call;
it does not silently manufacture duplicates or fall back to another method.
GPU libraries remain isolated; this adds no dependency or service.

For a mixed diagnostic, first restore only exactly associated render predictions:

```bash
uv run --locked --extra canonical python -m experiments.orientation.bind_probe \
  /path/to/views /path/to/probe-input-images /path/to/probe-output /path/to/new-oa
```

## Local artifacts and validation

Root: `data/experiments/runs/oa-distribution-fusion-01/`.

- `protocol.json`, `run.py`, `summary.json`: frozen settings, experiment driver and metrics.
- `{blue,red}-{sphere,ring}/`: bound OA outputs, full proposals, independently fitted
  even/odd results and decoded baselines.
- `split-comparison-axes.png`: the same display camera for baseline A, fused A and
  fused B; camera-space axis insets, not changed image geometry.
- `{blue,red}-agent/`: original model replies, events, evidence, direction checks,
  readable `trace.md` and final `work/agent-export-2-1/canonical.glb`.
- `{blue,red}-final-axes/`: all seven final camera views with XYZ overlays.
- `final-pair-axes.png`: three displayed camera directions for each mug.
- `source-snapshot/`: exact experiment/agent source files captured for the executed jobs.

Ruff, formatting, mypy and pytest cover the integration. Tests include known-pose
recovery, competing front retention, Euler poles, coordinate-basis changes,
flat/corrupt evidence, registered-render binding, refusal to promote unregistered
photos, preserved failures and rejection of a changed proposal method on resume.
Synthetic tests validate implementation behavior, not learned semantic accuracy.

Next evaluation should keep this route optional, expand the independent object
set and distinguish a pose-estimation conflict from competing semantic definitions.
More geometric agreement alone cannot decide between functional and presentation
fronts or authorize category-standard admission.
