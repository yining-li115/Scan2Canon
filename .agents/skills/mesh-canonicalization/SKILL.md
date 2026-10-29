---
name: mesh-canonicalization
description: Use Scan2Canon's geometry, rendered evidence and validated decision tools to turn an existing GLB into an instance semantic canonical proposal, or explicitly preserve unresolved axes. Use for mesh orientation work, not reconstruction or dataset ground-truth annotation.
---

# Mesh canonicalization

Work from the repository root. Read `docs/canonicalization-evidence-agent.md` for
the current OA evidence-agent route and `docs/canonicalization-agent.md` for the
optional geometry-only tools. The coordinate basis is right-handed, +Y up and +Z
reference front when established. There is no category direction table or lookup
action. Do not recreate one in prompts or infer a universal standard from one case.

## Evidence workflow

1. Verify input hashes, mesh appearance, render cameras and matching OA predictions.
   Preserve original photos as observations, not registered mesh-space directions.
2. Infer functional identity separately from appearance. State instance up/front
   hypotheses from visible evidence; keep meaningful alternatives or unspecified
   axes when the object does not support one interpretation. Labels do not determine
   direction. OA agreement and generated geometry can both be wrong.
3. Use the bounded evidence agent to inspect candidates and obtain materially
   different evidence through photographs, crops or additional rendered views.
   Support-plane refinement is a geometric proposal based on a uniform-density
   convex hull, not a measured center of mass or proof of semantic uprightness.
4. Let validated tools execute pose choices. Compare a refined candidate with its
   original parent. Never invent executable matrices, cite unseen evidence, exceed
   the job budget, or claim an unavailable tool succeeded. If observations cannot
   resolve the direction, preserve uncertainty or request a specific missing view.
5. Inspect the final manifest and canonical preview. A successful rigid export
   verifies geometry preservation, not semantic truth or cross-instance consistency.
   Keep contradictory descriptions, failures and partial-axis results in the record.

Record whether a result was produced by the independently executed model or by
this interactive visual agent. Do not silently alter a model decision after
review. Dataset admission and category-standard creation are separate tasks.

During inference, do not read scoring references, known random rotations, or
another policy's reference errors. Evaluate only after saving decisions. A change
made after seeing failures is development work and must not replace the original
run or count as an independent test. Historical jobs that used the removed table
remain historical; changed schemas, tools or prompts require a fresh job.
