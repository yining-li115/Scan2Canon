# Appearance-aware canonicalization

The evidence agent now requires stored color or texture evidence by default. It
preserves imported materials when rendering OA inputs and candidate poses, and
includes up to two verified original photographs in the first visual-model request.
Additional photographs remain available through `retrieve_photos`; they are not
projected onto the mesh or treated as registered camera poses.

```text
GLB with vertex colors or UV textures
  → known-camera appearance-preserving renders
  → OA V2 direction proposals
  → colored candidate renders + original RGB photographs
  → bounded visual evidence agent
  → canonical GLB with verified appearance preservation
```

Color can identify functional parts that geometry misses. Generated colors can also
be incorrect: the model is instructed to compare them with real captures, distinguish
functional identity from decorative shape, and preserve ambiguity when evidence
conflicts. Neither a logo nor the presence of texture establishes a universal front.

## Input requirements

`experiments.appearance.describe` inventories vertex/face colors, UV coordinates,
embedded images and material factors. It records whether there is color evidence
and whether actual texture images exist. It rejects malformed UVs or textures
without a usable UV mapping. A uniform neutral mesh has no appearance signal under
this check; a plain untextured white object needs an explicit shape-only run. This
is a conservative input gate, not a score of information content or reconstruction
fidelity. Materials or generated texture cannot substitute for missing real evidence.

The agent fails before dispatching a model request if appearance is absent. Use
`--allow-shape-only` deliberately for a geometry-only baseline. Supplying separate
photographs does not bypass this check or silently claim that the mesh is textured.
The older fixed-flow open worker remains available as a legacy comparison.

Use a verified observation bundle bound to the same mesh hash, following the
[existing manifest contract](canonicalization-agent.md#supplemental-photographs).
A bundle must actually contain photos of that instance; assigning a matching hash
alone does not establish identity. Keep raw captures unchanged. Photos used to
reconstruct the mesh are not held-out evaluation evidence.

## Prepare and run

Render a new bundle locally, keeping imported appearance:

```bash
uv run --locked --extra canonical python -m experiments.orientation.render \
  /absolute/path/object.glb /absolute/path/colored-views \
  --blender /Applications/Blender.app/Contents/MacOS/Blender \
  --views 12 --require-appearance
```

Run OA on the GPU server using those exact images:

```bash
experiments/orientation/.venv/bin/python -m experiments.orientation.infer \
  /absolute/path/colored-views /absolute/path/oa-results --require-appearance
```

OA receives the known-camera colored renders, **not the original user photos**.
The adapter preserves RGB and composites alpha on white. The input manifest binds
image, mesh and appearance hashes to the predictions. Any new render bundle needs
its own OA inference; never rewrite an old prediction's manifest hash to reuse it.
Existing predictions remain reusable only with their unchanged original bundle.
Legacy bundles without appearance metadata are inspected from their embedded mesh
when admitted to the agent; their original pixels and manifest remain unchanged.

Run the agent with original observations:

```bash
uv run --locked --extra canonical --env-file .env.local \
  python -m experiments.canonicalization.agent_worker \
  /absolute/path/colored-views /absolute/path/oa-results /absolute/path/agent-job \
  --observations /absolute/path/original-photo-bundle \
  --backend gemini --max-calls 4 \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
```

The first two photographs in bundle order remain visible alongside the initial
renders and candidates. Supplemental evidence uses at most the remaining slots
within the 32-image cap. Photo IDs cannot duplicate visible evidence through a
redundant retrieval. Final choices citing photos still require explicit visible
photo-to-mesh correspondence; no image-space direction is applied directly to an
unregistered mesh.

## Export and trace

A rigid export is checked against the input's color, UV, image and material
inventory, in addition to the inverse-transform geometry check. A mismatch fails
the stage instead of reporting success. This protects stored appearance; it cannot
recover detail missing from the reconstruction.

`request.json` records the appearance gate and input inventory. `events.jsonl`
records `initial_evidence` with the first-turn photo IDs and appearance inventory.
Each saved model request records the image hashes. New render manifests record
imported-material rendering and Blender material/color-attribute counts. Final
reports include the appearance inventory and `artifact.appearance_preserved`.

Changes to prompts, appearance checks and photo admission are versioned through
existing source hashes. Old completed jobs must not be resumed using new code and
presented as the original experiment; keep them and use a new run directory.

## Local validation on the existing SAM 3D result

`data/experiments/runs/appearance-integration-01/` contains:

- Twelve newly rendered colored views ready for a new OA GPU run.
- A mesh-bound original capture bundle.
- Fresh candidate renders using genuine cached OA predictions and their unchanged
  older colored source views, plus a preview of the first VLM request with the photo attached.
- A rotated GLB using a recorded prior proposal, with exact color preservation and
  an inverse-transform geometry check; this is not a new semantic decision.
- A synthetic texture fixture verifying UV/image preservation and chromatic pixels
  in real Blender output; it is explicitly test data, not a reconstructed object.

The SAM 3D model has vertex colors, not a UV texture atlas. Its subtle beige/gray
appearance and missing fine detail are properties of that reconstruction. The local
checks do not assert photorealistic appearance or improved orientation accuracy.
No new cloud-model request or GPU OA inference was dispatched for this validation.

Validation passed with Ruff, mypy and 204 pytest tests. Tests cover immediate photo
visibility, additional-photo retrieval and eviction under the 32-image cap, gray
input rejection before dispatch, strict OA input rejection before checkpoint load,
UV/image preservation through rigid export, and failure when export strips color.
The real Blender texture fixture retained a mean RGB channel spread of 65/255
among opaque rendered pixels; this is a diagnostic color-presence check, not a
photometric accuracy score.
