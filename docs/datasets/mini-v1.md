# Scan2Canon Mini v1

This pilot contains **37 distinct physical object instances**: 12 NOCS objects,
20 OO3D-9D objects, and 5 user-captured objects. It supports development and failure
analysis, not a statistically conclusive claim about open-world accuracy.

The versioned [catalog](../../experiments/datasets/mini-v1.json) fixes source IDs,
selection rules, partitions, available file hashes, and annotation requirements.
Videos, frames, rotations, and reconstruction attempts do not add object instances.
The authored chair from the earlier coordinate diagnostic is not included.

## Composition

| Source | Categories | Instances |
| --- | --- | ---: |
| NOCS `real_test` | Bottle, bowl, camera, can, laptop, mug; two each | 12 |
| OO3D-9D | Ball, bottle, bowl, chair, cup, handbag, mouse, scissor, speaker, toy truck; two each | 20 |
| Own captures | Hole punch, speaker, mug, chair, spray bottle | 5 |

The upstream category `cup` is preserved rather than silently relabeled as `mug`.
The OO3D-9D categories target direction cues, approximate symmetry, thin geometry,
and ambiguous resting poses. These are sampling motivations, not reviewed labels
for the selected instances. A ball, for example, may have appearance cues, and the
selected ball records have no explicit upstream symmetry fields.

Public instances are selected without running orientation models: sort each
category's candidates by the SHA-256 of
`scan2canon-mini-v1:17:<source-instance-key>` and take the first two. NOCS keys are
the full OBJ archive member names; OO3D-9D keys come from `name2oid.json`.
The catalog records the exact selected instances and source-index hashes.

## Partitions

- **Development: 16 public objects.** One instance per source/category pair for
  debugging, selecting rules, and setting provisional thresholds.
- **Holdout: 16 public objects.** The other instance in each source/category pair.
  Freeze rules and thresholds before examining model predictions on these objects.
- **Diagnostic: 5 own objects.** Test the capture-to-generated-mesh workflow. The
  hole punch was already used in development and is not an independent test case.

All images, rotated copies, crops, and generated meshes of an instance inherit its
partition. These are held-out instances within selected categories, not a test of
unseen categories. Upstream OV9D train/test membership is recorded separately;
membership in our holdout does not establish that OA, PointSO, or a VLM never saw
the asset in pretraining. Check overlaps before making novelty claims.

## Initial acquisition status — October 27, 2026

- **NOCS:** 12 OBJ meshes and 12 normalization sidecars downloaded from the official
  archive using HTTP byte ranges. ZIP member sizes and CRC32 values were checked,
  and SHA-256 hashes recorded. All meshes load with nonempty faces and finite
  vertices. Raw bytes and source coordinates are preserved.
- **OO3D-9D:** 20 instance IDs and their upstream symmetry records fixed; the name,
  scene-archive, and symmetry indexes downloaded. **Meshes are not downloaded.**
  The official downloader uses 100 mesh batches; the first is 2,653,351,491 bytes.
  A direct individual mesh URL returned 404. Locate the required batches before
  transferring bulk data; do not run the upstream downloader indiscriminately,
  since it also downloads checkpoints and scene data unrelated to this pilot.
- **Own:** all five original videos preserved locally, with extracted frames.
  The earlier generated hole-punch mesh is retained as a prediction. The other
  four objects have not been reconstructed.
- **References at acquisition:** no reviewed Scan2Canon reference frames. The
  subsequent [NOCS source audit and rotation preparation](../experiments/2026-10-27-nocs-preparation.md)
  records checked source conventions separately from this initial selection.
  A selected or downloaded asset is not automatically ready for accuracy scoring.

Local files live under `data/datasets/scan2canon-mini-v1/`. The machine-local
`acquisition.json` records availability separately from the versioned selection.
Source indexes and download/mesh checks are under `sources/`; downloaded NOCS
files are under `assets/nocs/obj_models/real_test/`. This data is Git-ignored.
On another machine, fetch the official archive and extract only catalog members,
or transfer these verified local files. Check the catalog hashes before use.

The NOCS OBJ files reference material files absent from the published archive's
real-test entries. UV coordinates alone do not provide the missing textures.
Render this group explicitly as **shape-only** unless matching source materials
are independently recovered; do not manufacture appearance. All 12 meshes are
non-watertight when loaded without processing/welding. Do not treat their raw
volume centroids as measured physical centers of mass. Any convex-hull or
uniform-density estimate must record its assumption and remain auxiliary evidence.

## Reference annotations and evaluation

The project convention is right-handed, with +Y up and +Z front. Audit each
source's axes and each category's meaning before assigning a source-to-project
rotation. Camera pose and an arbitrary object coordinate frame do not by themselves
establish a functional front. An absent symmetry field is unreviewed evidence,
not proof that a unique orientation exists.

Keep reference annotations separate from inference inputs. Reuse reviewed source
labels and record any category convention changes independently of OA, PointSO,
and VLM predictions. Preserve equivalent orientations when appropriate. For mugs,
specify the handle convention; for spray bottles, distinguish nozzle direction
from label-facing direction. User confirmation is reserved for unresolved rules
or missing evidence, rather than requested for every asset.

Run the public-mesh track independently of SAM 3D. Apply recorded random rotations
to the source mesh, render with known camera transforms, and measure recovery
against the reviewed reference. Rotation equivariance alone does not prove that
the recovered semantic direction is correct.

Run own captures as a separate end-to-end diagnostic: selected real frame and
mask, generated mesh, orientation candidates, and checks against unused real
views. A plausible generated surface is not ground-truth geometry. Do not pool
unannotated own captures into the public set's angular-accuracy denominator.

Report results separately by source and category: up error, symmetry-aware
orientation error where defined, coverage, error among automatically accepted
cases, unresolved cases, and latency. Count failures and abstentions explicitly.
Multiple rotations remain correlated measurements of one object. The small
holdout is useful for early regressions, not precise estimates of rare failures.
This catalog does not implement inference, acceptance calibration, or approval.

## Sources and use

- [NOCS official repository](https://github.com/hughw19/NOCS_CVPR2019), revision
  `dd58dbf68feede04c3d7bbafeb9212af1a43422f`: mesh archive and data-use terms.
  The README restricts the dataset to non-commercial use and requires citation;
  the software license does not override those dataset terms.
- [OO3D-9D / OV9D](https://github.com/caijunhao/ov9d), revision
  `d1065c3ffdf2c959f9b6a553f9af4820ae85cf81`: source indexes, BOP model records,
  and official download recipe. Its source objects are real scans; its scene
  images are rendered. Preserve original OmniObject3D and derivative terms before
  redistributing any assets.
- User captures remain local. This pilot does not publish videos, meshes, or a
  mixed-license dataset. Source IDs in the catalog identify evaluation inputs.
