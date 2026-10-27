# Resumable canonicalization worker

The standalone worker takes an existing GLB and an explicit category hypothesis,
then runs geometry generation, visual candidate selection and validated rigid
export. It replaces manual decision-file authoring with a model policy. It does
not reconstruct the object, infer an unknown category convention, approve dataset
admission, or connect the experiment to the capture HTTP API.

## Run with a cloud visual model

For Gemini, start with [environment setup](environment.md): fill `GEMINI_API_KEY`
in `.env.local`, keep `GEMINI_MODEL` configured, and use `--backend gemini`.
The worker uses Gemini's native `generateContent` endpoint. The OpenAI configuration
below is an alternative; the two providers use different credentials and adapters.

Run from the repository checkout with Python 3.12+, the root `uv.lock`, the
`canonical` extra and Blender. This route does not need CUDA or a rented GPU.

Configure `OPENAI_API_KEY` in the process environment, or create `.env.local` in
the repository root containing that variable. `.env.local` is ignored by Git.
Do not put credentials in commands, model prompts, run manifests or commits.

```bash
uv sync --locked --extra canonical
uv run --locked --extra canonical --env-file .env.local \
  python -m experiments.canonicalization.worker \
  /absolute/path/object.glb mug data/experiments/runs/object-canonical-01 \
  --backend responses --model gpt-6-astra \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
```

On Linux, use `--blender blender`. Omit `--env-file` when the key is already in
the environment. The model is explicit; there is no silent provider/model
fallback. The documented `gpt-6-astra` option supports image input, Responses and
structured outputs according to its [official model page](https://developers.openai.com/api/docs/models/gpt-6-astra).
Account access and task accuracy must be verified in a live run.

The adapter makes one Responses request per selection, with `store=false`,
medium reasoning, at most 8,192 output tokens (including reasoning), a 60-second
socket timeout and no transport retries. Input images are white-composited JPEGs
with a maximum side of 768 pixels, using high image detail. Raw source images
remain unchanged. It records the returned model ID, response ID and token usage.
A refusal, incomplete response, unknown candidate or invalid JSON is a failure,
not a semantic abstention. The current adapter expects a reasoning-capable model
supporting these settings. The timeout is not a guarantee that an interrupted
request was never processed or billed by the provider.

For a verified photo bundle, add `--observations /absolute/path/bundle`. It must
contain `manifest.json` in the format described in the
[agent tool guide](canonicalization-agent.md#supplemental-photographs), bound to
this exact mesh hash and category. Provide at most eight supplemental images so
the largest 12-candidate case still fits after 12 opposite views are added. The
controller rejects requests exceeding 32 images. It does not associate arbitrary
photos with objects automatically. Unregistered photos cannot replace mesh-space
candidate geometry.

A local Qwen adapter remains available through `--backend local --weights DIR`,
using the separately configured semantic-orientation GPU environment and verified
weight manifest. It is a distinct policy and has not been evaluated as a
replacement for the cloud policy in this worker.

## Decision flow and limits

1. Preserve the input mesh and verified observations in the run directory.
2. Generate up candidates from surface PCA and support planes; render them.
3. Ask the policy to select a candidate with visible part evidence or return
   `needs_evidence`. Candidate IDs are the only executable pose choice.
4. If uncertain and budget remains, render opposite views of the same candidates
   and ask once more. Additional views do not create new geometry or repair a scan.
5. Once up is selected, generate and select front candidates only when the
   category convention defines a front. Apply the same bounded evidence loop.
6. Validate hashes, candidate membership, rotations and up/front consistency.
   Export a rigid proposal and verify its inverse transformation and topology.

There are at most two selection calls per axis and four per job. `--max-calls N`
can lower the total to 1–4. Calls are journaled **before dispatch**, so timeouts
and process interruptions consume budget. Each geometry/export stage has at most
two attempts across restarts. The policy cannot invoke arbitrary tools, change
budgets or conventions, supply a rotation matrix, or grant dataset approval.

The current reviewed category definitions are `bottle`, `bowl`, `can`, `camera`,
`laptop` and `mug`. Bottle, bowl and can leave yaw unspecified. A plausible stable
support is evidence for an up candidate, not proof of semantic uprightness; the
support calculation uses a uniform-density convex-hull approximation.

| Result scope | Export | Worker state |
| --- | --- | --- |
| `full_frame` | Up and front proposal | `succeeded` |
| `up_only` | Up proposal; convention leaves yaw unspecified | `succeeded` |
| `up_only_front_pending` | Up proposal; front remains unresolved | `needs_evidence` |
| `unresolved` | No canonical mesh | `needs_evidence` |

Every scope retains `dataset_approval: not_granted`. A successful worker execution
is not a semantic correctness certificate. In `needs_evidence`, `state.json`
contains a missing-evidence description; a future application can use it to
request a targeted photo. This worker does not display a user-confirmation UI.

## Artifacts and recovery

The CLI prints the path of the completed `canonicalization.json`. Read
`state.json` for the result directory rather than assuming an attempt number.

```text
request.json              Input hashes, policy settings, code/convention/dependency versions
state.json                Stage checkpoints, attempts, result scope and missing evidence
calls/up-1.json            Prompt, image hashes, counted dispatch, raw reply and usage
calls/front-1.json         Separate front decision request, when needed
decisions/                Strict decisions bound to candidate and convention hashes
work/inputs-1/            Original mesh and verified photo snapshot
work/up-1/                Candidate poses, renders and Blender version
work/up-extra-1/          Optional opposite-view evidence
work/front-1/             Front candidates bound to the selected up decision
work/export-1/            canonicalization.json and canonical.glb when an axis is established
```

The report includes the transform, inverse, geometry checks, scope, actual model
call count, policy identity and request hash. Keep the whole run directory for
reproducibility, including all attempted stages. Never infer approval from a GLB
file alone: a failed export attempt may have written a file before validation.

After a transient failure, repeat **the same command** with `--resume`. A lock
prevents concurrent workers on the same run. Completed stages and responses are
verified and reused; unfinished tool attempts get a new directory. A finished
run resumes without new model calls. A provider call with no saved response can
be retried only within the original call limits. Invalid saved model responses
remain failures rather than triggering unlimited repair requests.

Keep the original input file, observation manifest, policy, budget, code and
locked environment unchanged for resume. Modified completed artifacts are
rejected. When adding evidence, changing the convention/model/code, or starting
another experiment after an exhausted budget, use a new output directory and
retain the previous run. Repeated fresh runs are separate experiments, not
free retries or independent evidence of accuracy. Model services can still be
nondeterministic or update aliases: checkpointed replay is reproducible, while
fresh cloud inference is not guaranteed to produce identical decisions.

## Code boundaries and validation

- `model_policy.py`: request/reply contract and the model boundary.
- `gemini_policy.py`, `responses_policy.py`, `local_policy.py`: concrete model adapters.
- `compute.py`: existing geometry/render/export tools plus supplemental evidence.
- `controller.py`: bounded execution, checkpoints, provenance and recovery.
- `runtime.py`: explicit composition; `worker.py`: thin CLI.
- `workflow.py`: existing decision and geometry validation, shared with manual use.

The test suite exercises rigid exports, partial axes, supplemental evidence,
invalid decisions, altered artifacts, interrupted calls, interrupted exports,
exhausted budgets and Responses payload/error handling. These tests use explicit
scripted model/render fixtures. They do not measure model accuracy.

A local Blender integration smoke test under
`data/experiments/runs/canonical-worker-smoke-01` uses a real NOCS mesh and previously
recorded decisions to exercise an extra-view round, rigid export and zero-call
resume. Its policy is recorded as `recorded_fixture_not_live_inference`. It is
software integration evidence, not a new holdout score. Cloud model evaluation
requires credentials and a separate run; no live cloud accuracy is claimed here.

API formats follow the official [image input guide](https://developers.openai.com/api/docs/guides/images-vision)
and [structured output guide](https://developers.openai.com/api/docs/guides/structured-outputs).

Gemini request formats follow the official [structured output guide](https://ai.google.dev/gemini-api/docs/generate-content/structured-output) and [image input guide](https://ai.google.dev/gemini-api/docs/image-understanding).
