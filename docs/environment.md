# Environment setup

The canonicalization worker supports Gemini, Claude, OpenAI Responses and a local Qwen
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

The command below is the fixed-category geometry baseline. The
[OA V2 open semantic worker](open-canonicalization.md#prepare-and-run) uses the
same Gemini environment variables, with a maximum of three calls instead of four.

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
The evidence-agent controller can switch models after a confirmed HTTP 429; this
is a separately counted call, not a hidden retry inside the adapter. Its reviewed
candidate list is `experiments/canonicalization/gemini-models.json` (override with
`GEMINI_MODELS_FILE`). `GEMINI_MODEL` chooses the first model; `--model` pins one
model and disables fallback. See [model fallback and correction](canonicalization-evidence-agent.md#model-fallback-and-correction).
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
- HTTP 429 can mean a rate limit or exhausted quota. The evidence agent tries
  the next configured model within its remaining budget; check AI Studio when
  the list is exhausted. Failed requests consume the worker's attempt budget. Changing
  API keys in the same project does not create a separate quota.

Sources: [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing) and
[rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).
Google's [data-use terms](https://ai.google.dev/gemini-api/terms) vary by tier and
region; free access does not by itself specify data handling.

Transport and configuration tests use mocked responses and do not consume quota.
A [first live Gemini diagnostic](experiments/2026-10-27-gemini-canonicalization.md)
completed one mug with two generation calls on a user-confirmed Free Tier project.
This verifies that path and one object, not general canonicalization accuracy.

## Configure Claude

The evidence, open semantic and geometry workers also accept `--backend claude`.
The Claude adapter uses the native Messages API with image blocks and schema-guided
JSON. No SDK, GPU dependency or new service is required. Keep the key only in
`.env.local`, and preserve existing Gemini configuration:

```dotenv
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-5-5
ANTHROPIC_STOP_AT=2026-11-05T00:00:00+00:00
```

`--model` overrides `ANTHROPIC_MODEL`; the default is `claude-sonnet-5-5`, verified
available to this account on 2026-10-29. The provider is always explicit: Claude
has no model fallback list and never switches to another provider. Changing the
provider, model or deadline requires a fresh job rather than resuming old work.

For an existing appearance-preserving mesh and matching OA predictions:

```bash
uv run --locked --extra canonical --env-file .env.local \
  python -m experiments.canonicalization.agent_worker \
  /absolute/path/views /absolute/path/oa data/experiments/runs/claude-object-01 \
  --observations /absolute/path/observations --backend claude --max-calls 4 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
```

### Promotional-credit safeguards

Promotional credits are a billing balance, not a permanently free API tier.
Confirm the remaining credit, eligible products and expiry in Claude Console.
The observed grant expires on November 5, 2026; the example deadline stops at
00:00 UTC that day, conservatively before any later expiry. It is a local guard,
not live credit verification. Set it for your own grant; leaving it empty disables
the date guard. The adapter checks it before both counting and generation.

Each generation has a free `count_tokens` preflight including the images and
schema included in the prompt. Missing/invalid counts or estimates above 50,000 block generation.
Output is capped at 8,192 tokens. The existing harness journals attempts before
calling the adapter, counts failures against the 2–4 call job budget, and refuses
blind retries after a lost response. Preflight is an additional HTTP request,
not another generation. Reports preserve returned usage and preflight counts.
There are no server tools, prompt caching controls, fast mode or automatic retries.

At the official Sonnet 5.5 rates checked on 2026-10-29 ($2/M input and $10/M
output), 12 generations at those limits estimate $2.18304; a $3 planning envelope
allows margin. Token counts are estimates, so this is not a provider-enforced
dollar ceiling. Limits apply per job, not to all programs sharing the key. Keep
purchased balance at zero and automatic recharge off when using promotional
credit only. This code neither purchases credits nor changes payment settings,
and cannot verify the billing balance, expiry or whether another process spends it.

Sources: [pricing](https://platform.claude.com/docs/en/about-claude/pricing),
[token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting),
[structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).
The full agent schema exceeded Claude's constrained-decoding grammar limit in the
first live trial. The adapter therefore supplies the complete original JSON schema
in the prompt and requests JSON text, without `output_config.format`. Local Pydantic
validation still enforces every bound and semantic contract before tool execution.
Malformed replies consume a call and return validation feedback within the same
budget; they are never executed or silently repaired.
