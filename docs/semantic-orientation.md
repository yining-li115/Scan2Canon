# Semantic front comparison

This experiment compares OA V2, rule-grounded VLM candidate selection, and PointSO
semantic direction prediction. It is separate from the production harness. A
successful inference or stable prediction never approves a dataset asset.

## Scope and conventions

The initial cases reuse the authored chair and generated hole-punch controls, each
before and after a recorded rotation. They test integration and equivariance, not
open-world accuracy. Only the chair has an authored canonical frame. The punch's
paper-entry-outward convention is a **draft**, not accepted ground truth.

Rules in `experiments/semantic_orientation/rules` specify the up direction, front,
required visible part, and PointSO queries. They are fixed before inference. The
punch front points OUT of the paper slot; paper insertion points the opposite way.

All input meshes, renders, points, rules, source revisions, weights, raw model
responses, and evaluated transforms are recorded or checksummed. No reference
frame or ground-truth rotation is included in model prompts. The evaluator reads
control annotations separately.

## Methods

- **OA raw medoid:** choose the predicted full frame minimizing total angular
  distance to the other views. This is an explicit comparison baseline, not a
  reproduction of OA's training-label aggregation or the existing symmetry solver.
- **OA + support:** retain its front but project it onto the selected support plane.
- **VLM + support:** present eight render views around the selected up axis to
  Qwen2.5-VL-7B-Instruct with the same category rule. Run normal and reversed image
  order. Accept a candidate for review only when both structured responses select
  the same ID and cite valid evidence IDs. Invalid JSON, missing evidence, and
  disagreement produce no selected pose. Textual evidence is not verified grounding.
- **PointSO raw / + support:** use the pinned official `pred_orientation` with one
  instruction per call and twelve votes; preserve all raw direction outputs. Query
  up and front independently. The support variant replaces predicted up with the
  shared geometric candidate. Near-parallel or nonfinite axes are rejected.

The pinned upstream PointSO helper repeats a list of multiple instructions while
reshaping its output in contiguous vote groups. Single-instruction calls avoid
mixing prompts without modifying upstream code. This is not a newly trained model.

All support variants share OA-ranked stable-plane up evidence. They are ablations
of front selection, not three fully independent end-to-end systems. VLM selection
has a 45-degree yaw grid (at most 22.5 degrees quantization to the nearest candidate
when the up estimate is correct); continuous methods have no such quantization.

Renders inspect the same mesh. They do not reveal new real evidence, and a generated
part can be consistently wrong. The current VLM experiment receives mesh renders
only. Real-capture grounding, automatic part back-projection, new-category rule
creation, and user evidence collection are not implemented here.

## Local preparation

Use the previously completed controls and OA runs described in
[Orientation diagnostics](orientation-diagnostics.md). Example for one case:

```bash
uv run --locked --extra canonical python -m experiments.semantic_orientation.prepare \
  data/experiments/inputs/orientation-controls-01/chair-base \
  data/experiments/runs/orientation-controls-01/chair-base \
  experiments/semantic_orientation/rules/chair.json \
  data/experiments/inputs/semantic-controls-01/chair-base \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
```

Repeat for chair-rotated, punch-base and punch-rotated, using `hole-punch.json` for
the punch. Preparation samples 30,000 surface XYZRGB points with seed 17, ranks
support planes, and renders eight views with known camera transforms. Output
coordinates stay tied to the original GLB; no silent camera-frame conversion is
applied to PointSO output.

## GPU execution

Commit local code first, pull the same commit on the server, and transfer the
ignored prepared inputs. Install models in the isolated Linux environment:

```bash
bash scripts/setup_semantic_orientation.sh
experiments/semantic_orientation/.venv/bin/python -m experiments.semantic_orientation.assets \
  pointso weights/semantic-pointso
experiments/semantic_orientation/.venv/bin/python -m experiments.semantic_orientation.assets \
  vlm weights/semantic-vlm
```

PointSO also loads the upstream CLIP ViT-B/32 asset; its official downloader checks
SHA-256 and each prediction run records that digest. Qwen uses local safetensors,
BF16 and SDPA. There is no model API key or external inference service.

```bash
experiments/semantic_orientation/.venv/bin/python -m experiments.semantic_orientation.pointso \
  data/experiments/runs/semantic-pointso-01 data/experiments/inputs/semantic-controls-01/*/
experiments/semantic_orientation/.venv/bin/python -m experiments.semantic_orientation.vlm \
  data/experiments/runs/semantic-vlm-01 data/experiments/inputs/semantic-controls-01/*/
```

Run sequentially so model memory is released between experiments. Download result
directories before local evaluation:

```bash
uv run --locked --extra canonical python -m experiments.semantic_orientation.evaluate \
  data/experiments/runs/semantic-pointso-01 data/experiments/runs/semantic-vlm-01 \
  data/experiments/inputs/orientation-controls-01 \
  data/experiments/runs/semantic-comparison-01 \
  data/experiments/inputs/semantic-controls-01/*/
```

## Evaluation and next dataset

Report chair full-frame, up and front errors; both objects' rotation equivariance;
VLM order agreement and abstentions; raw PointSO vote dispersion; and execution
cost. No symmetry discount is inferred from model answers. An abstained case is
unavailable, never scored as zero error. Agreement does not establish correctness.

A subsequent benchmark needs independently reviewed conventions and diverse real
objects: held-out categories, new instances, unusual subtypes, occlusion, and objects
without a unique front. Measure accepted-set error versus coverage, incorrect
automatic acceptance, user intervention rate, and latency. Two diagnostic objects
cannot determine which method generalizes best.

## References

- [SoFar / PointSO](https://qizekun.github.io/sofar/) and
  [official implementation](https://github.com/qizekun/SoFar).
- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).
- [One-shot 3D Object Canonicalization](https://jinli998.github.io/One-shot_3D_Object_Canonicalization/):
  category reference and semantic/geometric alignment.
- [CanoVerse](https://arxiv.org/html/2603.07144v1): support/semantic hypotheses and
  human candidate selection. Our support stage is not a full reproduction.
- [DORI](https://arxiv.org/abs/2505.21649) and
  [SPATIALUNCERTAIN](https://arxiv.org/abs/2605.30557): motivation to test orientation
  and evidence sufficiency separately from general visual understanding.
