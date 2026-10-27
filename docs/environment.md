# Environment setup

The canonicalization worker supports Gemini, OpenAI Responses and a local Qwen
policy. For Gemini, fill the key in `.env.local`; no OpenAI key or CUDA setup is
needed. Blender is still required for candidate rendering.

## Configure Gemini

1. Create a Gemini Developer API key in [Google AI Studio](https://aistudio.google.com/apikey).
   Use the project whose API tier and quota you intend to use.
2. Open `.env.local` at the repository root. If it does not exist in a fresh
   checkout, copy `.env.example` to `.env.local` first. Preserve any existing
   configuration rather than overwriting it.
3. Paste your key after `GEMINI_API_KEY=` and save. Keep `OPENAI_API_KEY` blank
   when using Gemini. The template selects `gemini-3.8-flash`; model access and
   quota must be checked for your project in AI Studio.

```dotenv
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.8-flash
```

The blank key above is intentional. Do not put the real key into `.env.example`,
chat messages, source code or commits. `.env.local` is ignored by Git. On macOS
or Linux, `chmod 600 .env.local` limits access to its owner.

The worker does not automatically read dotenv files. Pass `--env-file .env.local`
to `uv run`, or set the variables in the process environment. Existing process
environment values take precedence over values loaded by uv from the file.

| Variable | Used by | Meaning |
| --- | --- | --- |
| `GEMINI_API_KEY` | `--backend gemini` | Gemini Developer API key |
| `GEMINI_MODEL` | `--backend gemini` | Model ID, overridden by `--model` |
| `OPENAI_API_KEY` | `--backend responses` | Optional OpenAI key; not used by Gemini |
| `SCAN2CANON_DATA_DIR` | Capture backend | Capture/job storage; not the standalone worker's output directory |

The backend must be selected explicitly. There is no automatic provider switch,
model upgrade, or billing activation. The worker cannot inspect or enforce your
Google project's billing tier.

## Run a mesh

From the repository root:

```bash
uv sync --locked --extra canonical
uv run --locked --extra canonical --env-file .env.local \
  python -m experiments.canonicalization.worker \
  /absolute/path/object.glb mug data/experiments/runs/gemini-object-01 \
  --backend gemini \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
```

Replace the mesh path and category. On Linux use `--blender blender`. The output
must be a new directory. To resume after a transient error, repeat the exact
command with `--resume`. Changing the model requires a new run directory.

The controller allows at most four model calls per object, with at most two per
axis. `--max-calls 1` can limit an initial connectivity test to one selection,
but front may then remain unresolved. Refusals, incomplete responses and quota
errors do not produce a successful semantic result.

The Gemini adapter sends labeled JPEG images through native `generateContent`,
requests JSON constrained by the decision schema, and saves returned model
version and token usage. It uses the model's default thinking settings, an
8,192-output-token cap and a 60-second socket timeout. It does not automatically
retry network errors or rate limits. API key headers are not saved in artifacts.
See [worker behavior and recovery](canonicalization-worker.md) for partial axes,
validated geometry and provenance.

## Free quota and rate limits

Checked against Google's documentation on **2026-10-27**:

- Gemini Developer API has a Free Tier for selected models. The pricing page
  lists free input/output for `gemini-3.8-flash` on that tier; this does not make
  all models free or guarantee free requests on a paid project.
- Limits apply per project, not per key. Requests per minute, input tokens per
  minute and requests per day can each limit a run. Daily request quota resets
  at midnight Pacific time.
- Check your active model limits in AI Studio using the link on Google's rate
  limits page. Do not assume a fixed universal daily allowance.
- HTTP 429 can mean a rate limit or exhausted quota. Check AI Studio before
  resuming; failed requests still consume the worker's attempt budget. Changing
  API keys in the same project does not create a separate quota.

Sources: [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing) and
[rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).
Google's [data-use terms](https://ai.google.dev/gemini-api/terms) vary by tier and
region; free access does not by itself specify data handling.

Transport and configuration tests use mocked responses and do not consume quota.
Live Gemini connectivity and canonicalization accuracy have not yet been
verified; they require a configured key and a separate inference run.
