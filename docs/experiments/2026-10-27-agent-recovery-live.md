# Live run of model fallback and validation feedback

This development run uses the existing SAM 3D hole-punch mesh with vertex colors,
the original reconstruction photo, and the verified OA predictions from the
[appearance rerun](2026-10-27-appearance-live-rerun.md). It does not repeat GPU
inference or use independent pose ground truth.

## Reproduce

Run from the repository root with the user's confirmed Free Gemini project key
in the ignored `.env.local` file. Choose a new output directory for each new run.
The command below is the exact invocation for this experiment; an existing output
directory is deliberately not overwritten.

```bash
UV_CACHE_DIR=.uv-cache uv run --locked --extra canonical --env-file .env.local \
  env GEMINI_MODEL=gemini-3.6-flash \
  python -m experiments.canonicalization.agent_worker \
  data/experiments/runs/appearance-integration-01/oa-ready-views \
  data/experiments/runs/appearance-agent-live-01/oa \
  data/experiments/runs/appearance-agent-recovery-01/job \
  --observations data/experiments/runs/appearance-integration-01/observations \
  --backend gemini --max-calls 4 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
```

This invocation overrides the initial model for this process without changing
`.env.local`. It keeps the configured fallback chain enabled; `--model` would
instead pin a single model. The four-call limit includes rate-limited requests,
fallbacks, successful decisions and corrections. It is not a per-model budget.
The job's `request.json` records model identities, source versions and input hashes.

To generate a readable execution log without another model call:

```bash
UV_CACHE_DIR=.uv-cache uv run --locked --extra canonical \
  python -m experiments.canonicalization.agent_report \
  data/experiments/runs/appearance-agent-recovery-01/job
```

For intentional recovery, repeat the original command with `--resume` and all
inputs, budget, model order and source versions unchanged. A saved completed
response is reused. A dispatch without a response cannot be automatically replayed.

## Evaluation boundary

This run tests the real execution path. A switch is demonstrated only if a provider
429 actually occurs, and correction is demonstrated only if a rejected reply is
followed by a valid response. Neither is forced by consuming quota or altering
model replies. A valid export remains a pose proposal, not dataset ground truth.

## Observed result

The live worker completed successfully with `scope: full_frame` after two calls
to `gemini-3.6-flash`. The first call requested top views of candidates 1, 5 and 9.
Blender executed that action. The second call selected candidate 1, citing
candidate 5 as a contrast and the original photo as part-correspondence evidence.
The decision passed the unchanged evidence and export checks.

- Model request time in total: 50.19 seconds.
- Local tool-stage time in total: 9.80 seconds.
- Actual model switches: zero. Validation rejections: zero.
- Output vertices/faces: 316,954 / 633,888; scale unchanged.
- Inverse-transform vertex error: approximately 1.42e-8, below the 2.00e-6
  geometry tolerance. This is export preservation error, **not pose accuracy**.
- Vertex-color hash preserved; no UV texture atlas is present in this source.

Artifacts under `data/experiments/runs/appearance-agent-recovery-01/`:

- `job/work/agent-export-2-1/canonical.glb`: actual new exported mesh.
- `job/work/agent-export-2-1/canonicalization.json`: decision, geometry checks and
  actual decision-model identity.
- `job/trace.md`: readable record of actual calls and tool execution.
- `canonical-preview/contact-sheet.jpg`: six Blender views of the exported GLB.

The preview shows the handle above the base and retains the source's light/dark
colors. The base still appears somewhat tilted in the front/back views; there is
no independently annotated up/front direction to quantify pose accuracy. Front
and Top labels describe the selected coordinate frame, not certified semantics.
The output remains `dataset_approved: false`.

This validates the updated pipeline's normal successful path with real API calls.
Automatic quota switching and rejected-reply correction were not triggered here;
their previously passing tests remain simulated failure/recovery evidence.

A subsequent exact-command `--resume` reused both saved model responses and
completed stages. The number of dispatched model calls remained two, and the
same exported report was returned without further API requests or rendering.
