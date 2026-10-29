# Evidence-agent examples with execution traces

## Protocol

This is a development diagnostic of the current `oa-evidence-agent-v1` worker,
not a held-out accuracy estimate or an evaluation of newly trained models.

Four distinct NOCS objects were selected before inference: bowl, mug, camera and
laptop. Selection used the lexicographically first development case ID per category,
without consulting reference rotations or prior scores. Camera and laptop were
also run with identity-matched public RGB crops, giving six conditions in total.
The intuition that bowl/mug would be easier and camera/laptop harder was a hypothesis,
not a measured difficulty label. Labels and this hypothesis were not supplied to
the model. The agent can see the available pilot-convention keys.

The run reuses genuine cached OA V2 predictions, validated against each render bundle
and pinned checkpoint metadata. It freshly executes candidate rendering, Gemini
requests, selected evidence tools, decision validation and rigid mesh export.
It does not rerun OA inference or SAM 3D reconstruction. Input meshes lack texture;
original RGB is available only in the two photo conditions. No personal capture is
included in this cohort. Public source: Stanford NOCS `real_test.zip`.

- Model: the configured Gemini `gemini-3.8-flash`, using the user's confirmed Free project.
- Budget: four dispatches per case, at most 24 in total; no automatic model retries.
- Stop condition: first rate-limit response or two consecutive transport failures.
- Tool selection: autonomous; no forced action sequence or required coverage.
- State and trace: original responses, snapshots, checksummed artifacts and failures remain saved.
- Reference access: scoring only after the cohort's model decisions are saved.
- Photo conditions: camera `scene_1/0187`, laptop `scene_5/0265`; exact instance
  metadata and mask crops bind each image bundle to its input mesh hash. These are
  unregistered images, so the agent must explain visible photo-to-mesh correspondence.

Local artifacts live under:

```text
data/experiments/runs/evidence-agent-examples-01/
  selection.json          # frozen cohort and budget
  run-examples.py          # preserved local runner
  assess-examples.py       # separate post-inference assessment
  <condition>/events.jsonl # timestamped execution events
  <condition>/trace.md     # readable timeline
  <condition>/calls/       # actual requests and responses
```

`status: succeeded` means a validated export was produced. It does not mean that
semantic orientation is independently correct. `up_only` intentionally leaves yaw
unspecified; it is not a complete canonical frame. Every export remains
`dataset_approved: false`.

## Observed results

Five conditions were attempted. Two consecutive transport failures stopped the
batch before the sixth condition, as specified in the protocol. There were 15
model dispatches, 13 saved responses, three validated exports, and no automatic
retry or paid-provider fallback.

| Condition | Executed evidence actions, in order | Final outcome | Up error | Frame error |
| --- | --- | --- | ---: | ---: |
| Bowl, shape only | convention → top render | Up-only proposal | 9.80° | Not defined |
| Mug, shape only | convention → top render | Up-only proposal | 5.56° | Not proposed |
| Camera, shape only | convention → right render | Full-frame proposal | 31.57° | 35.63° |
| Laptop, shape only | top render → right render | Third model request failed in transport | — | — |
| Camera, RGB available | convention → retrieve photo | Third model request failed in transport | — | — |
| Laptop, RGB available | None | Not started after batch stop | — | — |

Errors compare saved output rotations against audited NOCS source-coordinate
conventions, **not independently collected human ground truth**. The 15° threshold
used in earlier project diagnostics would accept the bowl/mug up directions and
reject the camera direction. No aggregate success-rate estimate is justified by
these four development objects, paired conditions, and incomplete responses.

The bowl correctly preserves unspecified yaw. The mug identifies the handle and
retrieves the handle-based convention, but its selected upright candidate has OA
fit 0, so the harness does not permit a full frame. The model chooses `up_only`
while explicitly reporting missing yaw evidence. The camera exports a frame but
is insufficiently accurate: successful execution alone does not validate orientation.

The laptop's two saved intermediate responses remain uncertain and describe an
open hinged case/container. That is a useful identity failure signal from the
untextured mesh, not a completed final classification. Its RGB condition has not
run. The camera's photo was actually retrieved, included in the subsequent model
request, and saved with provenance; the missing response means this run cannot
establish whether appearance improved the decision.

## Candidate availability diagnostic

Only after inference, reference rotations were used to measure the best possible
error among the retained candidate pool. These are oracle diagnostics, not agent
scores and not feedback to the model.

| Input | Best available up error | Best available full-frame error |
| --- | ---: | ---: |
| Bowl | 1.90° | Not defined |
| Mug | 5.56° | 89.97° |
| Camera | 31.57° | 35.63° |
| Laptop | 90.07° | 101.34° |

The camera selected the best retained candidate under both reference metrics,
but that candidate is still poor. The laptop pool contains no suitable reference
up direction at all. These numbers describe the **retained** pool, not every raw
OA prediction before filtering. Ranking improvements alone cannot repair these
cases. The bowl, in contrast, had a more accurate up candidate than the selected
one, leaving room for selection or geometric refinement.

## Tool and recovery coverage

| Action | Autonomous real-model evidence in this cohort |
| --- | --- |
| `lookup_convention` | Executed 4 times |
| `render_view` | Executed 5 times using real Blender |
| `retrieve_photos` | Executed once; one public RGB crop supplied to the next request |
| `inspect_image` | Not selected; covered by controlled behavior tests |
| `request_capture` | Not selected; pause/submission/resume covered by controlled tests |
| `finish` | Selected 3 times, each followed by validated rigid export |

No tool was forced merely to increase coverage. The latter two actions have no
new autonomous live-model demonstration in this cohort. Existing tests do not
establish their learned selection quality.

A real completed bowl job was resumed using the same arguments. All three saved
responses and five completed stages were reused, with **zero new model dispatches**.
The trace summary deduplicates observed evidence across this replay. Transport
failures remain counted and cannot be silently replayed.

The sequential run took approximately 8.9 minutes. About 8.2 minutes were spent
inside model calls (including failed requests), versus about 38 seconds in tool
stages. The adapter uses a 60-second socket timeout; both failed calls took about
61 seconds. This identifies model transport/latency as a separate operational
issue from candidate quality; a missing response is not an orientation error.

## Review artifacts and validation

Open `data/experiments/runs/evidence-agent-examples-01/review.html` for the comparison
and expandable action arguments. `comparison.jpg` shows actual re-rendered exports;
failed rows show an explicitly unaccepted candidate, never a replacement output.
`assessment.json` preserves output errors, candidate oracle diagnostics, reference
hashes, final decisions, and executed actions. Each attempted condition has a
`trace.md`, `events.jsonl`, and `trace-summary.json`.

Ruff, format checking, mypy and the full pytest suite passed: **198 tests**.
New tests verify trace replay counts, actual versus rejected actions, secret-free
transport error logging, and call-budget preservation after failure.

Next work should address candidate coverage/refinement and then rerun the unfinished
appearance comparison under a separately versioned protocol. The completed cohort
and its failures should remain unchanged. More agent roles or more tool calls alone
would not solve the observed absence of accurate pose candidates.
