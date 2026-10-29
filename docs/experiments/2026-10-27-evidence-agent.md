# Evidence agent implementation validation

Date: 2026-10-27. The [evidence agent guide](../canonicalization-evidence-agent.md)
describes the bounded model/action/observation loop and CLI capture continuation.

## Automated verification

Ruff lint/formatting, strict mypy and all **196 tests** passed. Twenty agent tests
exercise chosen camera views, crops, photo retrieval and convention lookup;
invalid/repeated actions; terminal budgets; model transport failures; tool and
capture-copy recovery; pending-task continuation; and wrong-mesh or modified
capture rejection. Captured photos can be retrieved again after context eviction,
without resetting the task budget or repeating earlier model calls.

The policies and test renders are explicitly scripted fixtures. These tests
measure execution behavior and safeguards, not semantic accuracy, autonomous
planning quality or manual-review reduction.

## Real Blender integration

The existing source mesh/render bundle and saved real OA outputs were used:

- `data/experiments/inputs/video-5573-orientation-02`
- `data/experiments/runs/orientation-video-5573-01`

`evidence-agent-integration-01` exposed a relative-path failure: Blender's renderer
returned an absolute directory while the action code retained a relative output
path. That failed run and its call record were preserved. Evidence output paths
are now normalized; a regression test exercises a relative CLI output path and
an absolute renderer result.

`data/experiments/runs/evidence-agent-integration-02` then completed these scripted
actions with real Blender and PIL:

1. Render candidates 0, 1 and 2 from behind.
2. Crop the returned back view of candidate 1.
3. Export candidate 0 as an up-only proposal for mechanical verification.

The selection was made by `integration_fixture / scripted-not-semantic-inference`,
not Gemini and not an independent visual assessment. Candidate 0 is not certified
as correctly oriented. The inverse vertex error was 3.67635e-9 in mesh units.
The resulting view contact sheet was inspected for successful rendering. Repeating
the run with resume reused its three saved replies and completed tool artifacts.

No new GPU inference or checkpoint download was needed. There is no claim of
improved canonicalization quality from this integration test.

## Live Gemini action selection

`data/experiments/runs/evidence-agent-gemini-01` uses Gemini 3.8 Flash with a two-call
budget and the same saved OA/mesh inputs. No category label or evaluation reference
was supplied. The first live reply identified a two-hole paper punch and selected
`render_view` for candidates **1, 5 and 6** from **top**. Its stated purpose was to
distinguish the operator-facing side from the hinge side and inspect base-plate
alignment. These are the model's proposed interpretation and evidence goal.

The harness validated the IDs, rendered the requested views with Blender and
included those actual tool observations in the second request. This confirms a
live model-selected evidence action executed through the loop; it does not by
itself establish a correct final orientation or an accuracy improvement.

The second live reply finished with candidate **1**, `scope: full_frame`, citing
primary candidates 1 and 5, their requested top views, and source views of the
base. It proposed the operator-facing handle edge as front. The harness accepted
the structurally valid decision and exported a rigid canonical proposal:

`data/experiments/runs/evidence-agent-gemini-01/work/agent-export-2-1/canonical.glb`

The complete live run used **two Gemini calls**. Inverse vertex recovery error was
1.41501e-8 against a 1.99825e-6 tolerance in mesh units. All 316,954 vertices and
633,888 faces were preserved. These numbers verify the rigid export only; they
are not angular errors or independent evidence of semantic correctness. The
output remains `dataset_approved: false` and the original observations were not
replaced. A same-command resume reused the saved replies and tool artifacts.

This single example establishes execution of a learned action-selection loop.
It does not establish improved pose accuracy, cross-category consistency, or a
lower human-review rate. Those require controlled evaluation against the earlier
fixed-flow routes on the same cases and independent references.
