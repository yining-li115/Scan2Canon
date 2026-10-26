# Semantic front comparison — October 26, 2026

## Outcome

The three model routes executed on an A100 40 GB, using the same four prepared
cases: an authored chair and the generated hole punch, each at its original pose
and after a known rotation. No object was approved for the dataset.

Support-constrained OA and PointSO produced accurate chair frames in this small
control. PointSO's own up/front predictions conflicted. Qwen's choices depended
on image presentation order for the chair and it abstained on the punch. The
experiment supports retaining external validation and evidence requests; it does
not establish a general method ranking or a reliable open-world front detector.

## Reproduction and artifacts

Follow [Semantic front comparison](../semantic-orientation.md). Implementation and
initial environment: `8ec63681a6f4793b5752b4a5a6bfbb4233fbbd55`; completed PointSO
runtime: `0b549c9`; VLM format correction and axis guard: `2d58650`.

- Inputs: `data/experiments/inputs/semantic-controls-01/`.
- PointSO: `data/experiments/runs/semantic-pointso-02/`.
- Qwen: `data/experiments/runs/semantic-vlm-02/`.
- Evaluation and GLBs: `data/experiments/runs/semantic-comparison-01/`.
- Comparison figures: `chair-base-comparison.png`, `punch-base-comparison.png`.
- Original controls: `data/experiments/inputs/orientation-controls-01/`.

All raw prediction files, including the two initial diagnostic runs, were copied
back locally and matched server SHA-256 values. `local-verification.json` records
them. Model weights remain on the GPU server and can be downloaded again from the
pinned revisions. The server was left running, with 4 MiB GPU use and 0% utilization
after inference. Stopping or releasing it is a separate action.

## Frozen model inputs

The chair rule defines front as the direction a seated person faces. The punch
rule proposes front as horizontally outward from the paper-entry slot, opposite
to paper insertion. The latter remains a draft; no human front label was supplied.

Each PointSO input is 30,000 surface XYZRGB points, with twelve official sampled
votes per query. Queries were fixed before inference: `top` for up; `the direction
a seated person faces` or `outward from the paper insertion opening` for front.
Different wording may change results and would require a separate prompt ablation.

Qwen received eight known-camera ring renders and the category rule, once in each
image order. It did not receive real photos, ground-truth rotations, or other
models' output labels. Both orders must select the same candidate ID before a
pose can be retained for review. None passed that condition.

## Accuracy on the authored chair

Full-frame angular error in degrees, with no symmetry discount:

| Method | Original chair | Rotated chair |
| --- | ---: | ---: |
| OA raw medoid | 12.40 | 3.00 |
| OA + support | 3.51 | 2.89 |
| PointSO raw | Rejected: conflicting axes | Rejected: conflicting axes |
| PointSO front + support | 0.38 | 4.51 |
| Qwen selection + support | No consistent selection | No consistent selection |

These are two poses of one simple procedural chair, not two independent real
objects. PointSO is not uniformly better than OA, and the support variants share
the same OA-ranked up estimate.

PointSO's predicted up/front separation was 171.9 degrees on the original chair
and 138.2 degrees on the rotated chair. A canonical up/front pair should be near
90 degrees. Replacing predicted up with the support direction made its front
usable on this fixture.

## The generated hole punch

There is no accepted front ground truth, so no absolute front accuracy is reported.

- OA + support returned frames differing by 5.64 degrees after undoing the known
  input rotation. This shows consistency for this aggregation, not correctness.
  It is a raw-frame medoid baseline and differs from the earlier symmetry-mode
  candidate selection and support preview yaw.
- PointSO's front was only 6.4 / 2.5 degrees from the selected up direction. Projecting
  such a nearly vertical vector onto a horizontal plane would amplify small errors
  into an arbitrary front. Both cases were rejected by the semantic-axis guard.
- PointSO's twelve front votes were tightly clustered (median deviation about
  1 degree), illustrating that repeatability alone does not validate semantics.
- Qwen requested evidence in both orders and both poses, saying it could not
  identify the paper-entry slot. That is the model's statement, not a verified
  finding that the slot is absent. The system retained the request and no pose.

The next evidence should concern the real slot and its correspondence to the mesh.
Additional renders of the same generated surface cannot establish that a generated
feature is physically correct.

## Qwen order sensitivity

All eight responses in the second run satisfied the structured contract:

| Case | Normal image order | Reversed image order |
| --- | --- | --- |
| chair-base | Requests more evidence | Selects candidate 0 |
| chair-rotated | Selects candidate 7 | Selects candidate 0 |
| punch-base | Requests more evidence | Requests more evidence |
| punch-rotated | Requests more evidence | Requests more evidence |

A single Qwen2.5-VL-7B configuration cannot establish the limits of all VLMs. In
particular, the chair abstention was overly cautious: the fixture's backrest and
seat are visible. Refusal is not automatically correct uncertainty estimation.
The order check catches an observable inconsistency but does not prove that an
order-consistent response would be correct.

## Diagnostic fixes and timing

The first PointSO launch failed before inference because upstream imported
`termcolor`, which was missing from the isolated runtime. The dependency was
added locally, locked, committed, and pulled before the successful run.

The first Qwen run revealed two contract issues: valid abstentions used an empty
`visible_evidence`, while some selected answers encoded evidence IDs as strings.
The contract now allows empty visible evidence only for non-selections, and the
prompt explicitly requires integer evidence IDs. First-run raw answers were
preserved; no silent coercion or repeated sampling until success was used.

After inspecting the first PointSO outputs, a provisional guard was added to
reject up/front separations outside 60–120 degrees. All model directions and votes
remain available. This guard is an engineering diagnostic, not a learned or
benchmark-calibrated threshold. It can reduce coverage and needs wider evaluation.

| Run | Measured elapsed | Model load | Peak allocated GPU memory |
| --- | ---: | ---: | ---: |
| PointSO, four cases / eight queries | 8.94 s | 7.20 s | 1.75 GiB |
| Qwen, four cases / eight responses | 28.90 s | 5.35 s | 19.76 GiB |

Times exclude input preparation, rendering, checkpoint verification, transfer, and
queueing. PointSO initialization includes its first CLIP download; Qwen had cached
weights. These are not app end-to-end latency estimates.

Local checks passed: 88 tests, Ruff, formatting, and mypy. GPU integration was
verified by actual execution; no checkpoint was installed into the backend environment.
