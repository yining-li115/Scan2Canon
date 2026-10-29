# Fresh GPU reconstruction and OA evidence-agent trial

The two existing mug captures were rerun on an A100 40 GB using code commit
`3f49638`. Both produced fresh colored meshes, verified OA predictions and
agent-selected **up-only** exports. Neither produced an established full frame.
This is a development diagnostic on familiar objects, not a held-out benchmark.

## Executed route

1. Reuse the preserved single-image RGB/mask bundles. Masks came from the earlier
   SAM 2.1 run with manually annotated boxes; capture extraction and segmentation
   were not rerun here.
2. Run the pinned SAM 3D model with seed 42. Preserve the GLB, Gaussian PLY,
   raw mesh arrays, predicted layout, package versions and checkpoint hashes.
3. Render twelve appearance-preserving views with recorded cameras in Blender.
4. Run pinned OA V2 with `--verify-official --require-appearance`. Compare the
   adapter against upstream `inf_single_case` on every image, including the raw
   900-dimensional outputs and decoded angles/symmetry fits.
5. Save the existing solver's modal-fit consensus proposal as the pure-OA baseline.
   Separately run the current evidence agent with all OA proposal modes, the
   colored candidate renders and two original photographs per object.
6. Let the agent select bounded evidence tools and export only validated axes.
   Render the baseline and actual exports with calibrated XYZ overlays.

The agent used Claude `claude-sonnet-5-5`, with four calls available per object.
It ran locally; API credentials were not copied to the GPU server. This trial
did not load the earlier pair-derived shared-frame proposal or a category table.
OA was a completed upstream stage, not a dynamically callable agent tool.

## Results

| Measurement | Blue faceted mug | Red-handle mug |
| --- | --- | --- |
| SAM 3D generation, excluding load | 19.20 s | 18.34 s |
| SAM 3D peak allocated memory | 17.89 GiB | 17.77 GiB |
| OA official parity | Passed, 12/12 images | Passed, 12/12 images |
| OA fits 0 / 1 / 2 / 4 | 8 / 0 / 3 / 1 | 7 / 2 / 2 / 1 |
| Pure-OA solver result | Needs evidence | Needs evidence |
| Agent calls | 2 | 2 |
| Executed evidence tool | `refine_up` | `refine_up` |
| Selected refined candidate / parent | 13 / 4 | 12 / 1 |
| Up correction relative to that parent | 8.74 degrees | 6.41 degrees |
| Export | Up-only | Up-only |
| Appearance preserved | Yes | Yes |

The correction angles are changes to the selected parent, not errors against
ground truth. The baseline pictured in the overview is the solver's leading
consensus representative, which need not be that parent. Its yaw and the agent's
representative yaw can differ; neither is asserted as a resolved front.

Both agents identified base-to-opening as up. Both retained handle-side and
logo/display-face interpretations as competing fronts. The blue mesh lacks the
logo visible in the real photos. The red agent's first interim assessment cited
hypothesis IDs as if they were image IDs; the harness recorded the assessment as
deferred and allowed its valid refinement action. The final assessment passed
the strict export checks. No rejected assessment authorized an export.

The final projected up checks were 0.34 degrees (blue) and 0.21 degrees (red),
with conditional angle intervals of 0–8.88 and 0–10.62 degrees. These are 2D
compatibility checks using the policy's own estimated landmarks, not independently
measured 3D accuracy. Rigid inverse-vertex errors were below 3e-16 in the internal
calculation. Geometry preservation does not establish semantic correctness.

Four generation replies used 90,899 input and 8,550 output tokens in total.
No provider fallback, billing-setting change or manual pose correction occurred.
The initial blue local attempt stopped before dispatch because its input transfer
was incomplete; its failure log is retained and it incurred no model call.

## Reproduction control and limitations

Before the fresh reconstructions, OA was also rerun on the two unchanged historical
render bundles. All 24 images passed official parity, and all decoded predictions
matched the historical run. Their fit counts were 9/0/2/1 (blue) and 6/3/2/1 (red).
This checks reproduction on identical inputs; it is not additional accuracy data.

Fit 0 means unresolved azimuth under OA's decoder. It does not prove physical
rotational symmetry, especially when a handle is visible. The fresh reconstruction
route still did not establish a shared horizontal direction. It also did not test
recognition/retrieval, independent reference annotations or repeated random input
rotations. No result was admitted as a category standard or dataset ground truth.

The next development question is how to preserve and select a reusable direction
meaning while respecting OA's learned frame. Repeated visual inspection alone
cannot choose between valid task definitions. Any such change should be evaluated
in a fresh run against this saved baseline, without silently relabeling its axes.

## Local artifacts

All experiment assets are under `data/experiments/runs/server-20261029/`:

- `comparison-axes.png`: pure-OA proposal versus actual agent export for both mugs.
- `results-summary.json`: metrics, hashes, selected candidates and check details.
- `existing-mesh-reproduction.json`: unchanged-input control results.
- `{blue,red}-mug-new/sam3d/`: complete generation assets and provenance.
- `{blue,red}-mug-new/oa/`: fresh predictions, logits and official parity results.
- `{blue,red}-mug-new/per-view-axes/`: independent predictions on each source view;
  axes remain explicitly unverified and the raw model images are unchanged.
- `{blue,red}-mug-new/agent/`: raw replies, tools, state, direction checks and traces.
- `{blue,red}-mug-new/agent-final-axes/`: six views of each up-only result.
- `local-backup-verified.json`: confirmation that both generated asset sets and
  OA/render bindings passed local checksum checks.

SSH bulk transfers intermittently timed out. Successful SCP transfers and bounded
read-only chunk transfers recovered the remaining assets; original SHA-256 hashes
were checked before considering the backup complete. Checkpoints remain on the
server and were not copied to the Mac. The server was not stopped.

Ruff and mypy passed; all 266 tests passed. No inference implementation or validation
gate was changed during this trial. Private captures and generated assets remain
outside Git; this document records their local locations.
