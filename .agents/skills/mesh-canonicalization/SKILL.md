---
name: mesh-canonicalization
description: Use Scan2Canon's geometry, rendered evidence and validated decision tools to turn an existing GLB into a semantic canonical mesh, or explicitly preserve unresolved axes. Use for mesh orientation work, not reconstruction or dataset ground-truth annotation.
---

# Mesh canonicalization

Work from the repository root. Use the existing tools in `experiments/canonicalization/`.
Read `docs/canonicalization-agent.md` for commands and the decision contract.

The convention is right-handed, +Y up and +Z reference front. Category definitions
are in `experiments/canonicalization/conventions.json`; they contain no instance
coordinates. An unsupported or uncertain category needs evidence, not a new
unreviewed category rule inferred from one object.

## Evidence workflow

1. Verify the input bundle. Generate upright candidates with `upright.prepare`.
   Inspect the rendered contact sheet and individual views when details matter.
   Support probabilities assume a uniform-density convex hull, not measured mass.
2. Identify observable parts that establish up. Compare alternatives explicitly.
   OA consensus may be consistently wrong; it is a proposal source. A scan hole
   is not automatically a mouth. For a laptop, establish which panel is the
   keyboard/base using recognizable details or a clear thickness distinction;
   a plausible open-laptop silhouette alone does not distinguish lid from base.
3. Submit a strict `workflow.Decision` bound to the case and convention hashes.
   Cite selected and contrasting image IDs and visible evidence. Use
   `needs_evidence` if the defining parts cannot be established. Do not supply an
   arbitrary rotation matrix, claim calibration from agreement, or call an
   unobserved tool successful.
4. If the convention leaves yaw unspecified, finish with up-only scope. Otherwise
   use `workflow front`, inspect those views, and submit a separate front decision.
   Select the direction facing the camera, not the view displaying the most parts.
5. Use `workflow finish`. It validates evidence, candidate membership, the up/front
   relationship, declared call limits and exported geometry. Read its final
   manifest, then inspect a canonical preview. Preserve failed runs and raw input.

Use at most two selection rounds for up and two for front. Extra rounds should
obtain a materially different view or geometry check, not repeat a confidence
question. Stop with an actionable missing-evidence description when the budget
is exhausted. Do not ask the user to approve every object.

Record the actual actor: `interactive_visual_agent` for this conversation's visual
reasoning; `local_vlm` for the independently executed model. These are distinct
experimental policies. Local Qwen results do not represent this agent's results.
The workflow is an experiment toolset, not a deployed iOS backend or independent
semantic certification. Dataset admission remains separate from generating a
canonical proposal.

During inference, do not read scoring references, source coordinate annotations,
known random rotations, or another policy's reference errors. Use the evaluator
only after saving decisions. A correction after seeing errors is development
work and must not replace the original result or count as an independent test.
