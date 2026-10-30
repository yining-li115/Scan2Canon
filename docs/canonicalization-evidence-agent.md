# Canonicalization evidence agent

This experimental worker runs **one visual agent**. On each turn, the model sees
the available mesh evidence and tool history, then chooses a typed action. The
harness validates that action, executes a bounded tool, and returns its observation
on the next turn. It does not split identity and orientation into independent agents.

The [open semantic worker](open-canonicalization.md) remains the fixed-flow
comparison. Both routes reuse verified OA predictions, appearance-preserving
rendering, model transports, call journaling, input snapshots and rigid export.
Neither is wired into the capture HTTP API or iOS app yet.

An optional `--proposal-method distribution` route fuses the saved OA angle
marginals across registered cameras before this same agent loop. It runs on CPU,
retains multiple proposals and does not bypass any semantic checks. The default
remains `decoded`; switching requires a fresh job. See the
[fusion experiment](experiments/2026-10-29-distribution-fusion.md) for its mixed
split-view results and the two actual up-only agent outcomes.

Fusion v2 makes search initialization independent of input-list order and retains
per-view score correspondence. See the [top-1 stability trial](experiments/2026-10-30-top1-stability.md)
for its measured differences and remaining limits. It does not change this agent's
decision policy or add an automatic acceptance threshold; changed code needs a fresh job.

The [appearance-aware path](appearance-canonicalization.md) now rejects missing
mesh appearance by default and supplies original photos on the first turn. Use
`--allow-shape-only` explicitly for the older gray-mesh baseline.

## Available actions

| Action | Model chooses | Harness executes and checks |
| --- | --- | --- |
| `retrieve_photos` | 1–4 existing photo IDs | Loads only photographs from the mesh-bound, checksummed observation bundle |
| `inspect_image` | A visible image ID and normalized crop bounds | Crops existing pixels, records source hash and pixel bounds; adds no generated detail |
| `render_view` | Exactly three candidate IDs and back/left/right/top | Renders those candidates with a known camera and records candidate/view correspondence |
| `refine_up` | Exactly three original candidate IDs | Proposes bounded support-plane corrections, preserves parent yaw availability, and renders new candidates for inspection |
| `request_capture` | A specific question and the evidence it should reveal | Persists a pending request and pauses; does not send a message or operate the phone |
| `finish` | A validated full-frame, up-only or unresolved decision | Validates evidence and candidate membership, then exports a proposal when allowed |

Every action includes a reason; nonterminal actions also require expected evidence. The agent may revise its
instance identity hypothesis after observing new evidence. Function and decorative
shape remain separate fields. No fixed category label is required as input.

The current route (`oa-evidence-agent-v4`) adopts OA V2's learned coordinate frame
as its reference. The agent reviews the OA proposals against visible part relations,
rather than independently choosing a preferred functional or presentation front.
There is no category direction table, lookup action or category-name menu. Another
sensible display direction can remain an unselected hypothesis without vetoing the
adopted reference. Conflicting OA branches, unsupported part relations or contradictory
observations still require further evidence or a partial/unresolved result.

Requests and final reports retain `orientation_reference`, including the upstream
axis mapping and `category_annotation_verified: false`. A model-relative proposal
does not certify the category's training labels, semantic correctness or cross-instance
agreement. Historical v1–v3 jobs are preserved and cannot resume with the changed code
or action schema. Start a new job using the same raw input bundles. Library import
continues to accept completed v3 jobs without retroactively assigning this reference.

The [OA-reference trial](experiments/2026-10-30-oa-reference-review.md) records the
source research, a failed first attempt, improved validation feedback and two final
up-only results. The revised framing has not yet established full-frame agreement.

## Direction meanings and projected part checks

Every agent turn now contains `directions`, separate from its identity description:

- `hypotheses`: named up/front hypotheses, each with a role (`upright`, `functional`,
  `presentation`, `structural`, `model_reference`), a visible `from_part → to_part` definition and
  image citations. These role names are not a category list or priority ordering.
- `issues`: `missing_observation`, `competing_definitions` or
  `conflicting_observations`, each attached to an axis. Competing definitions need
  at least two explicit hypotheses. Under v4 these are conflicts within the adopted
  OA reference; a separate possible task definition alone is not an unresolved issue.
  A `model_reference` hypothesis still needs visible part anchors and the same checks.
- `symmetry`: an unverified yaw hypothesis (`unknown`, `none`, `discrete`,
  `continuous`), with geometry/appearance basis, evidence and counterevidence.
  It never authorizes symmetry reduction or proves equivalence of poses.
- `measurements`: policy-estimated image points for the hypothesis's start/end
  parts, a location uncertainty radius, candidate ID and exact image ID.
- `selected_up` / `selected_front`: hypothesis IDs, not executable rotations.

The harness executes a deterministic check on each submitted measurement. For a
known candidate or detail render it projects that candidate's +Y or +Z through
the observed start point using the saved render camera and compares this with the
observed start-to-end direction. This uses the pinhole projection tangent;
positive point depth cancels from the projected direction. The tool accounts for
endpoint uncertainty and its effect on the projection direction.
The resulting angle interval is conditional on the policy's stated location
radius; it is not a calibrated confidence interval or pose accuracy bound.

The output is `compatible`, `conflict` or `indeterminate`. The current 20-degree
2D tolerance, minimum separation of 4% of image width, four-uncertainty-radius
separation guard and near-viewing-ray guard are development settings, not
calibrated semantic accuracy thresholds. A nearly depth-aligned axis, close or
uncertain points, or missing camera calibration cannot pass. Original photographs
and crops remain unregistered/unsupported for this computation; they can still
provide identity and part evidence. The first version supports square render
images and does not triangulate 3D landmarks or register original photos.

An exported axis must have an issue-free selected hypothesis and at least one
compatible check on the chosen candidate, with no conflicting checks for that
hypothesis/candidate in the submitted assessment. Full-frame export needs both
axes; up-only needs up and leaves `selected_front` null. A rejection returns the
numeric results to the same agent within the existing call budget. The agent can
inspect another view, revise its proposal or return unresolved. The harness does
not choose a semantic definition or silently repair an axis.

The landmarks are estimates from the same policy that proposes the pose. A
passing check proves only **2D compatibility conditional on those estimates**;
it does not independently validate part localization, functional meaning, 3D
pose accuracy, cross-instance agreement or physical symmetry. Correspondingly,
reports explicitly set `semantic_correctness_verified` and `symmetry_verified`
to false. No category-standard admission occurs.

Inspect `<job>/direction-checks/agent-N.json` and the adjacent `agent-N/check-*.png`
overlays. A/B mark the actual policy points and their uncertainty circles; the
blue line is the projected positive axis. These audit overlays are generated
after inference and are not fed back as visual evidence. Numeric results are
recorded in `events.jsonl`, `trace.md` and the final result. Raw model replies
and failed checks remain available. The full structured assessment and final
checks are included in `canonicalization.json`.

### Evidence requests versus export checks

Evidence actions can leave measurements empty and selected axes null. The harness
still validates the action, available resources, semantic image citations and
execution budget. A malformed interim direction reference is recorded as
`interim_direction_deferred`; it contributes no passing checks and cannot authorize
an export, but it does not prevent a valid observation tool from executing. The
raw assessment and error remain in the history. Every final pose choice retains
the full direction/candidate/citation/export checks.

For unchanged image bytes, the same candidate, axis and part relation retain the
largest previously checked uncertainty radius within the job. Changing hypothesis
IDs or point positions does not lower that floor. Reports preserve both the
model's `reported_uncertainty` and the effective `uncertainty`. Resume rebuilds the
same history from saved replies. This is a conservative development guard against
self-tightening after numeric feedback, not independent or calibrated localization.
Different images can provide new evidence; repeated calls alone do not.

See the [second shared-frame trial](experiments/2026-10-29-evidence-recovery.md): both
objects executed refinement and supplemental views, but neither produced an
accepted full frame in that bounded run.

## Harness responsibilities

The controller owns execution. The model cannot choose file paths, shell commands,
rotation matrices, cloud providers or unlimited tool calls. Candidate and image IDs
are validated locally even when supplied as enums in the response schema.

- The default budget is four model calls for the entire task, including any calls
  after a user capture. `--max-calls 2` or `3` lowers it. Waiting for a user does not
  reset the budget.
- There is one action per call. At most three evidence actions can execute before
  the last call. A tool request on the last call stops as `budget_exhausted`; the
  harness does not execute work whose result cannot be inspected within budget.
- Repeated evidence actions are rejected, including reordered lists of the same
  render candidates or photos. A crop must retain at least 1% of its source area.
- At most one `refine_up` action can execute per job. It adds up to three candidates
  without replacing the original poses. Each correction is limited to 25 degrees
  and requires a contact polygon covering at least 10% of the projected footprint
  within a 0.3%-of-bounding-diagonal contact band. Competing support normals may
  cause refusal. These are conservative development guards, not calibrated accuracy
  thresholds. The agent must see the rendered correction and cite its original
  parent before choosing it. The correction projects the parent front onto the new
  horizontal plane; unresolved yaw remains unresolved. Stable resting poses do not
  establish semantic uprightness, and convex-hull density is not measured mass.
- A checksummed `support-assessment` stage flags original candidates with a usable
  correction larger than 3 degrees. Selecting a flagged original requires first
  calling `refine_up` with its ID. The agent may retain that original, but must cite
  its rendered refined alternative when one exists. This gates inspection, not
  acceptance of the geometry proposal. Unsupported geometry records an unavailable
  assessment instead of inventing a plane. The 3-degree gate is a development
  trigger, not a measured accuracy guarantee.
- The initial known-camera views and primary candidates stay visible. Up to two original photos also stay visible from the first turn.
  At most eight supplementary images occupy remaining slots within the 32-image
  cap, with older ones evicted from model context. Raw evidence remains on disk. At most 32 images are sent.
- Each dispatched call is recorded before transport. A missing response after a
  network error remains counted and is not automatically retried. Invalid replies
  remain on disk; validation feedback lets the agent correct its answer on the next
  budgeted call. No rejected tool or export executes. The last invalid reply stops
  the task as failed, with no export.
- Completed tool artifacts are checksummed. An interrupted tool gets at most one
  further attempt on resume; its model decision is reused.
- Photo crops retain their photographic provenance. A final decision citing photos
  or their crops must explain visible correspondence with the mesh and also cite
  primary candidate comparisons. Photographs remain unregistered.
- OA fit 0 cannot establish yaw, and a full frame needs resolved instance semantics
  with a unique proposed front. Rigid exports preserve geometry and appearance.
  Every output remains `dataset_approved: false`.

This is a short, bounded evidence loop. It has no unrestricted planner, long-term
learned memory, automatic reconstruction repair, or independently calibrated
semantic confidence. It uses versioned prompts and typed tools directly; it does
not dynamically load repository skills or require an MCP server.

The support-plane tool is a local numerical up refinement, not a reconstruction
repair or continuous semantic front estimator. It leaves geometry, scale and
appearance unchanged. Its guarded proposals and rejection reasons are saved in
the evidence stage; new candidate IDs are restored from that checksummed stage
on resume. See `up_refinement.py` for the geometry and `agent_tools.py` for rendering.

## Run

Prepare known-camera renders and matching pinned OA inference as described in the
[OA guide](canonicalization.md). For saved results, local execution needs Blender
and the root locked CPU environment; no new GPU inference is required.
Configure the selected cloud provider through [environment setup](environment.md).

```bash
uv run --locked --extra canonical --env-file .env.local \
  python -m experiments.canonicalization.agent_worker \
  data/experiments/inputs/video-5573-orientation-02 \
  data/experiments/runs/orientation-video-5573-01 \
  data/experiments/runs/hole-punch-agent-01 \
  --backend gemini --max-calls 4 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
```

Use a new output directory. On Linux use `--blender blender`. Optionally supply
`--observations /absolute/path/original-photo-bundle`. The first two photos in bundle order are attached immediately and remain visible.
The agent sees all available photo IDs and can retrieve additional photos. It cannot find arbitrary user files.
The OpenAI Responses adapter is an explicit alternative backend; no provider switch
occurs automatically.

## Model fallback and correction

For the Gemini evidence agent, `gemini-models.json` supplies an ordered candidate
pool reviewed against Google's Free-tier pricing on 2026-10-27. `GEMINI_MODEL`
selects the initial model; the remaining pool entries follow in file order. Without
that variable, the first entry is used. Set `GEMINI_MODELS_FILE` to an alternate
JSON file containing one to four distinct model IDs, or use `--model MODEL_ID` to
pin one model and disable fallback. Existing fixed-flow workers remain single-model.
The file is not live quota detection and does not change project billing.

Gemini's transport schema retains types, required fields and enums but omits numeric
bounds and array-length bounds from its constrained-decoding grammar. The complete
Pydantic contract still enforces those bounds locally before any action. This
avoids the schema rejection observed when adding nested direction evidence;
it does not weaken execution limits or enable automatic HTTP 400 retries. See
[Google's structured-output limitations](https://ai.google.dev/gemini-api/docs/structured-output?hl=en).

- Only an explicit provider HTTP 429 triggers a switch. This can indicate a
  short-term rate limit or quota exhaustion; it does not prove a daily limit.
- Each model is tried at most once in the chain after receiving 429. Switching
  preserves evidence and tool history and consumes the same total call budget.
  The chain does not cycle or switch keys/projects/providers. Exhausted models or
  budget stop the run. Shared project limits may affect every listed model.
- HTTP 400/401/403, blocked/incomplete responses, timeouts and uncertain transport
  failures do not trigger a switch. Diagnose those failures separately.
- A typed response or evidence-validation error goes back to the current model
  on its next budgeted call. The agent must inspect contrasting evidence, not
  simply append arbitrary IDs. Validation remains unchanged and cannot certify
  that the visual interpretation is correct.
- Confirmed 429 records and saved replies replay without new API calls on resume.
  An unconfirmed dispatch remains counted and cannot be automatically retried.
  Changing model order, model identity, source versions or budget requires a new job.

For example, with `GEMINI_MODEL=gemini-3.8-flash`, the default chain is
`gemini-3.8-flash → gemini-3.6-flash → gemini-3.5-flash-lite`. A four-call budget
includes failures, successful decisions and corrections together; it is not four
calls per model. No attempt is made to exhaust real quotas for testing.

The trace records `model_dispatched.model`, `model_switched.from_model/to_model`,
`model_fallback_stopped.reason` and safe `agent_decision_rejected.reason` feedback.
The final report's `decision_policy` identifies the model that actually finished;
`policy` records the initial model and configured fallback pool.

## User capture and continuation

A `request_capture` result has `status: needs_evidence`, `stop_reason:
awaiting_capture` and a `pending_capture` record containing the question, expected
evidence, mesh hash, turn and request ID. It exports no mesh for that pending choice.
Inspect the printed `canonicalization.json` and show its question in a future UI;
the current worker does not send it to a person automatically.

Prepare new photos using the existing [observation manifest contract](canonicalization-agent.md#supplemental-photographs),
bound to the same mesh. Append them to the pending task:

```bash
uv run --locked --extra canonical --env-file .env.local \
  python -m experiments.canonicalization.agent_worker \
  data/experiments/inputs/video-5573-orientation-02 \
  data/experiments/runs/orientation-video-5573-01 \
  data/experiments/runs/hole-punch-agent-01 \
  --backend gemini --max-calls 4 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender \
  --resume --additional-observations /absolute/path/new-photo-bundle
```

Keep the original arguments unchanged, including `--observations` if used initially.
The harness verifies the new bundle, copies it into a new immutable stage, binds it
to the pending request, and resumes after the saved model decision. Earlier reports
and raw responses remain intact. Without new evidence, resuming a pending task
returns the same request without spending another model call. Wrong-mesh evidence,
changed photos and replacement submissions are rejected. An exhausted task cannot
be given extra calls through a capture submission.

## Artifacts and implementation

The CLI prints `work/agent-export-N-*/canonicalization.json`. The report contains
current semantics and choice, actual policy identity, tool history, stop reason,
pending capture (if any), and optional rigid export metadata. `calls/agent-N.json`
preserves the request descriptor and raw response. `state.json` records calls,
checksummed tool attempts and capture submissions.

- `agent_contracts.py`: action arguments, semantic/pose reply contract and prompt.
- `agent_tools.py`: bounded evidence execution and provenance.
- `agent_controller.py`: model/action/observation loop and capture continuation.
- `agent_worker.py`: CLI; concrete implementations are composed in `runtime.py`.

Tests cover model-selected camera/crop/photo/refinement actions, invalid action
rejection, repeated actions, transport/tool failures, capture copying failures,
same-task continuation, wrong-mesh/modified evidence, and budget preservation.
Scripted policies test behavior only. A successful tool run or rigid export does
not establish learned decision quality or reduced human-review rates.

See the [validation record](experiments/2026-10-27-evidence-agent.md) for test
coverage, real Blender integration and the distinction between scripted and
learned decisions.

## Inspect execution logs

Every new job appends timestamped JSON records to `events.jsonl`. Events distinguish
model dispatch, saved/reused responses, proposed/validated/rejected actions, tool
start/completion/failure/reuse, new observations, capture submission and stop reason.
Model and tool durations are measured separately. Image IDs connect each call to
its evidence. Error events record safe error types and categories, not arbitrary
exception messages or credentials.

Generate a readable timeline and machine-readable summary without model calls:

```bash
uv run --locked --extra canonical \
  python -m experiments.canonicalization.agent_report /absolute/path/job
```

This writes `trace.md` and `trace-summary.json`. The summary counts dispatched
calls from `state.json` and deduplicates observed evidence stages across resumes.
A proposed action is not counted as executed evidence. `finish` and
`request_capture` are terminal decisions, listed separately from evidence tools.
The trace is diagnostic; checksummed state and artifacts remain authoritative.
Malformed trace lines are counted, not treated as successful execution. Historical
jobs without traces cannot acquire an execution history retroactively.

The [six-condition development diagnostic](experiments/2026-10-27-agent-example-runs.md)
records actual autonomous tool coverage, transport failures, retained-candidate
limits and post-inference orientation errors.

Dynamic image IDs are listed in each request's context rather than repeated as
large enums in nested response-schema arrays. The controller validates membership
before any action or export. This keeps supplemental views from inflating provider
grammar complexity without weakening local execution checks. HTTP failures are
logged as `http_400`, `http_403`, etc.; quota failures retain `rate_limit`.

## Claude transport

Select `--backend claude` explicitly to run the same agent contracts and geometry
tools through Claude Messages. See [environment setup](environment.md#configure-claude)
for key configuration, the credit-expiry guard and token limits. The adapter
changes neither the semantic validation gates nor the execution/recovery budget.
Original photos and render evidence are sent to the selected cloud provider.
