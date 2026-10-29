# Claude direction-evidence trial — 2026-10-29

## Question and inputs

Can Claude run the existing category-free evidence agent, use appearance and
original photos, recover from rejected decisions, and produce checked axis proposals?
This is a three-object development trial, not an independent accuracy benchmark.
The user authorized sending the two mugs' and spray bottle's photos and rendered
mesh evidence to Anthropic, consuming promotional credit without purchasing credit.

Inputs are unchanged from the Gemini direction-evidence trials:

- Blue mug: `data/experiments/runs/own-appearance-01/mug`.
- Red-handled mug: `data/experiments/runs/mug-consistency-5654-01`.
- Spray bottle: `data/experiments/runs/own-appearance-01/spray-bottle`.

The runner records input hashes, code versions, policy identity and execution
budgets in its protocol. Prior model decisions and evaluation labels are not
supplied to the policy. No hand-edited pose, point or semantic label is inserted.

## Transport failure and repair

`data/experiments/runs/claude-direction-evidence-01` preserves the initial HTTP 400
failure. The batch stopped on the first blue-mug attempt. Explicit diagnostics
showed that token counting accepted the actual request (20,790 estimated input
tokens), but generation rejected the compiled schema as too large. A small smoke
schema succeeding earlier did not establish support for the full agent contract.
One separate diagnostic generation attempt reproduced the rejection. No canonical
mesh was produced by that run. Free counting diagnostics and provider errors are
preserved alongside the failed job; they are not successful agent decisions.

The adapter now includes the full original JSON schema in the text prompt and
requests one JSON object without a native constrained-decoding grammar. Local
Pydantic validation, image/candidate membership checks, numeric direction checks,
execution limits and export gates remain unchanged. Invalid replies can be corrected
within the same job budget; no rejected action executes. The adapter identity
records `schema_prompt_local_validation`. There is no implicit retry or fallback.

## Fresh protocol

`data/experiments/runs/claude-direction-evidence-02/run.py` runs Claude Sonnet 5.5
through the existing `oa-evidence-agent-v3` controller. Each object has at most
three generation attempts, nine total for the repaired batch. Including the two
rejected generation attempts, this stays within the originally proposed 12-attempt
envelope. The previous Gemini runs allowed four attempts per object, so budget
parity is not claimed. All use the same semantic and numerical validation gates.

A free token preflight blocks estimated inputs above 50,000; output is capped at
8,192. The local deadline is 2026-11-05 00:00 UTC. At the rates checked on October
29 ($2/M input, $10/M output for Sonnet 5.5), nine capped generations estimate
$1.63728. This is a planning estimate, not provider-enforced dollar accounting.
No recharge or payment setting is changed. Returned usage is recorded per call.

Sources: [Claude pricing](https://platform.claude.com/docs/en/about-claude/pricing),
[token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting),
[structured-output limitations](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).

## Results

All three cases exported up-only proposals with preserved mesh appearance. No
user confirmation, manual pose edit, or new category convention was used.

| Object | Returned model replies | Executed evidence tools | Result | Estimated usage cost |
| --- | ---: | --- | --- | ---: |
| Blue mug | 2 | `refine_up` once | Up-only, candidate 12 | $0.137134 |
| Red-handled mug | 3 | `refine_up` once | Up-only, candidate 12 | $0.193248 |
| Spray bottle | 3 | `refine_up` once | Up-only, candidate 12 | $0.202908 |

The red mug's first action was rejected because its direction issue cited direction
hypothesis IDs where visible image IDs were required. The next reply corrected the
references; only then did refinement execute. The spray bottle's first reply was
invalid JSON (`expected ',' or '}'`); the second supplied a valid refinement action
and the third finished. Neither rejected reply executed a tool. These are real
recovery observations, not scripted policy fixtures. They also demonstrate that
Claude output needs local validation even when it sounds plausible.

Returned usage totals 177,430 input tokens and 17,843 output tokens, with zero
reported cache-write/read tokens. At the recorded rates this estimates **$0.53329**.
That is not a refreshed billing balance. The two earlier HTTP 400 generation
attempts returned no usage and are recorded separately, rather than assigned
invented token counts or declared free. Free token-count requests are not model
replies. No billing setting or payment method was changed. The local environment
now includes the explicitly selected model and November 5 UTC stop guard.

## Post-inference review

`audit.json`, `comparison.json`, `landmark-checks.png` and `canonical-results.png`
are under `data/experiments/runs/claude-direction-evidence-02`. `review.py` and
`compare.py` regenerate the audits from saved artifacts without provider calls.
Per-object `job/trace.md`, `events.jsonl`, raw `calls/agent-N.json`, direction-check
overlays, exported GLBs and six-view canonical previews remain local.

Both mug checks use a visible base-to-rim relation. The spray check uses base to
sprayer-head top. The overlays show the actual model points and uncertainty radii;
none were edited. The mugs' head points sit near the upper wall/rim, so the words
“cavity” or “mouth” in the model's description should not be taken as independent
proof that the cavity center was localized. These are approximate visual cues.

All three exported GLB hashes match the earlier Gemini results exactly, and their
computed up-axis differences are zero. **Agreement is not accuracy**: both policies
selected from the same finite OA/support-refined candidate pool. This trial shows
no pose improvement over Gemini and supplies no independent ground-truth error.

The mugs retain handle-vs-graphic ambiguity; the spray bottle retains nozzle-vs-label
ambiguity. No functional/presentation priority was imposed, so a unique front was
not exported. More credit or a different model alone does not resolve an unspecified
semantic objective. Single-view landmark compatibility also leaves 3D ambiguity;
these checks remain `semantic_correctness_verified: false` and do not approve
objects for the dataset.

## Validation

After the transport repair: Ruff check/format and mypy passed; all **249 tests**
passed. Tests cover token-preflight rejection, no automatic retries, local expiry,
secret-free records, image encoding, model composition and strict local validation.
The successful trial used the revised adapter; the failed native-grammar attempt
was preserved rather than overwritten. No Git commit or push was made.
