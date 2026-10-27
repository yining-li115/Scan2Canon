# Mesh-to-canonical pilot: 27 October 2026

The tool chain now produces actual canonical GLBs from existing meshes, with
separate up/front decisions, bounded semantic selection, explicit abstention and
inverse-transform geometry checks. It is an experiment workflow operated by the
interactive visual agent, not an autonomous iOS/backend integration.

## Inputs and protocol

NOCS provides 12 selected meshes across six categories. Each has five prepared
random rotations. We ran OA on all 30 development rotations, then evaluated the
interactive visual agent on the first rotation of each development object and
the first rotation of each of six different holdout objects. The remaining 24
holdout rotations were not evaluated in this initial experiment; they are covered
in the later [rotation-consistency experiment](2026-10-27-rotation-consistency.md).
The selected OO3D-9D objects and five user
captures are not covered by these results.

These meshes are gray, untextured scans, often with missing bottoms. The scoring
references combine audited NOCS source coordinates with the known input rotation.
They are not independent human semantic annotations. Repeated rotations share
geometry and must not be treated as independent objects.

The holdout protocol was saved before inspecting its candidate images, at commit
`317f4f1`. It records selected cases and hashes of the skill, conventions, geometry
tools, exporter and renderer. Decisions were saved before scoring. The later
direct-GLB CLI convenience path was checked against the already evaluated bottle
and produced identical candidate rotations; it did not change holdout decisions.

## Three distinct policies

1. **OA V2 baseline:** official single-image inference on 12 renders, transformed
   back to the input mesh frame. Compare a raw rotation medoid with support/PCA
   evidence. All 30 development cases ran on the A100. An additional bottle run
   verified official single-case parity and identical per-view predictions.
2. **Local Qwen2.5-VL-7B:** six development objects, two candidate presentation
   orders each. All 12 calls executed, but all six objects abstained. Responses
   sometimes failed to recognize parts that were visible in the candidate images.
3. **Interactive visual agent:** inspected geometric candidates, cited selected
   and contrasting images, submitted strict decisions, and called the validated
   export tools. These results must not be attributed to the local 7B model.

OA performed poorly under these particular gray-scan/random-rotation conditions:
raw up was within 15 degrees in 1/30 cases; support-assisted up in 3/30. Neither
produced a full frame within 15 degrees in the 15 directional cases. This is a
failure of this tested configuration, not a general benchmark claim about OA.
Support voting alone did not rescue consistently incorrect semantic predictions.

## Interactive-agent results

| Group | Objects | Up proposals | Up within 15° | Full-frame proposals | Full frame within 15° | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 6 | 5 | 4 | 3 | 2 | 1 |
| Holdout | 6 | 4 | 4 | 1 | 1 | 2 |

The development laptop was a failure: up error 48.99° and full-frame error 178.87°.
The gray open-laptop silhouette allowed the agent to confuse the lid and base.
The original proposal and scoring result are preserved. It is not an approved
canonical result. The skill was revised before holdout to require identifiable
panel evidence instead of accepting a plausible silhouette.

Holdout details:

| Category | Output scope | Up error | Full-frame error | Evidence outcome |
| --- | --- | ---: | ---: | --- |
| Bottle | Up only | 1.15° | Not defined | Narrow neck above body |
| Bowl | Up only | 0.30° | Not defined | Open rim above smaller base |
| Can | Up only | 2.67° | Not defined | Intact lid with off-center opening above body |
| Mug | Full frame | 0.29° | 6.18° | Open rim upward; handle direction defines +Z |
| Camera | Unresolved | — | — | Top controls/base insufficiently visible |
| Laptop | Unresolved | — | — | Keyboard/base cannot be distinguished reliably from lid |

No per-object human confirmation was requested during either interactive run.
Unresolved cases contain missing-evidence requests rather than forced exports.
This small test supports a working workflow, not a calibrated accuracy guarantee
or a claim of broad category generalization. In particular, only one directional
holdout object received a full-frame proposal.

## Export validation and artifacts

The exporter preserves scale and topology counts, applies only a proper rigid
transform, and checks the reloaded GLB against the source using the inverse
transform. Up-only results explicitly leave yaw unspecified. Geometry checks do
not certify semantic correctness; dataset admission remains ungranted.

Raw outputs are local ignored data, with all completed-stage artifacts preserved:

- `data/experiments/runs/nocs-development-oa-01/`: 30 OA cases and raw logits.
- `data/experiments/runs/nocs-oa-parity-01/`: official entrypoint parity.
- `data/experiments/runs/nocs-development-evidence-01/`: geometry diagnostics and baseline scores.
- `data/experiments/runs/nocs-upright-vlm-development-probe-01/`: all local Qwen replies.
- `data/experiments/runs/nocs-agent-development-probe-01/`: six decisions, five exports, including the retained laptop failure.
- `data/experiments/runs/nocs-agent-holdout-01/`: frozen protocol, six decisions, four exports, scores and `comparison.jpg`.
- `data/experiments/runs/nocs-direct-entry-smoke-01/`: direct GLB command smoke test.

[Machine-readable results](2026-10-27-canonicalization-results.json) retain the
summaries, per-object errors and source-manifest hashes in Git. Large meshes,
renders and checkpoint files remain outside version control.

## Implementation and next boundary

See [the executable workflow](../canonicalization-agent.md) and
[the repository skill](../../.agents/skills/mesh-canonicalization/SKILL.md).
No model SDK was added to the domain or backend runtime, and no new dependency or
service was required. The GPU environments retain their existing lockfiles.

A deployable version still needs a model-backed policy adapter that can execute
the skill's decisions with comparable reliability, central call accounting,
and broader evaluation. Additional real photographs or textured meshes are
particularly relevant for the ambiguous camera/laptop cases. Repeating the same
confidence question does not supply that missing evidence.
