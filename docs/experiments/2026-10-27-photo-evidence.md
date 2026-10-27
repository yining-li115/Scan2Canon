# Supplemental-photo diagnostic: 27 October 2026

Adding identity-matched photographs and opposite-side candidate renders resolved
the previously abstained camera. The laptop remained unresolved. This is a
diagnostic follow-up on two known failures, not a new independent holdout score.

| Instance | Gray-only result | Supplemented result | Up error | Full-frame error |
| --- | --- | --- | ---: | ---: |
| `camera_shengjun_norm` | Unresolved | Full-frame proposal | 2.23° | 5.90° |
| `laptop_air_xin_norm` | Unresolved | Unresolved | Not measurable | Not measurable |

Both camera errors meet the existing 15° threshold. The old decisions and the
30-case rotation-consistency report are preserved unchanged. This run uses only
the first prepared rotation of each of these two instances; it does not establish
five-rotation recovery for the camera or raise the earlier benchmark's coverage.

## Sources and selection

The [official NOCS repository](https://github.com/hughw19/NOCS_CVPR2019) links the
real-test image archive. HTTP byte-range access indexed the 3,614,434,858-byte
archive without downloading it in full. Six small initial scene metadata records
identified scenes containing the exact catalog instances. Five selected frames
provided 15 RGB/mask/metadata files, totaling 3,174,855 uncompressed bytes; ZIP
member lengths and CRC32 were checked and SHA-256 values recorded.

Frame selection used available RGB filenames, not pose/error labels:

- Scene 2: first, middle and last available frames (`0000`, `0291`, `0629`),
  containing both instances.
- Scene 5: middle frame `0265`, containing the camera.
- Scene 1: middle frame `0187`, containing the laptop.

Each instance therefore has four supplemental crops. Official instance masks
and metadata were used to associate objects and crop the RGB evidence; this is
dataset-provided identity/segmentation assistance, not an app's predicted mask.
The [pinned upstream loader](https://github.com/hughw19/NOCS_CVPR2019/blob/dd58dbf68feede04c3d7bbafeb9212af1a43422f/dataset.py)
reads the red mask channel (OpenCV BGR channel 2). The local loader handles it
through PIL RGB channel 0, including indexed PNGs. Crops retain a 12-pixel margin.
No coordinate maps, depth files or ground-truth pose files were downloaded.

The images remain local under the source dataset's terms. The Git report contains
IDs, hashes, decisions and measurements, not redistributed source photographs.

## Procedure and observations

The existing upright candidates and their rotations were copied unchanged into a
new run. The protocol records prior decisions, source identities, mesh hashes and
tool versions. One full RGB scene and the earlier failures were known before the
protocol was written. New candidate decisions were saved before scoring.

Opposite-side renders of every existing candidate were added while inspecting
the supplemental evidence. The initial photo manifest was retained; the final
manifest additionally binds those renders. This was a **joint photo-plus-view
intervention**. The experiment cannot isolate the gain from photos alone, and
the alternate view itself may account for some of the improvement.

The camera photographs show a portrait instant camera with its circular lens low
on the body and its flash/ejection area above. The opposite render of upright
candidate 07 reveals a matching low circular projection and upper band; candidate
06 puts the lens high. The subsequent front candidate 00 exposes that circular
lens toward the viewer, whereas candidate 01 is the plain back. Both selected
rotations came from the existing geometric tools, not an estimated photograph
camera pose or an invented rotation matrix.

The laptop photographs clearly distinguish the black display from the silver
keyboard/trackpad base. The scan still offers two nearly featureless panels and
rough boundaries. Both open-laptop placements remain plausible without a reliable
part-to-surface match. The missing evidence is now a registered RGB-to-mesh view
or a mesh landmark identifying the keyboard/base panel. No canonical GLB was
exported for the laptop, and no per-object human confirmation was requested.

The camera used two cumulative up-selection rounds (the previous abstention plus
this round) and one front round. The laptop used two cumulative up rounds. Both
remain within the existing budget. These are interactive-agent decisions, not
results from the local Qwen policy or a deployed app backend.

## Artifacts and validation

Raw data are under `data/experiments/runs/nocs-photo-evidence-01/`:

- `protocol.json`, `selection-audit.json`: provenance and pre-scoring decision audit.
- `case-*-up/observations/`: original photographs, masks, metadata, crops and extra renders.
- `case-*-prior-up.json`: unchanged earlier decisions.
- `case-*-up.json`, camera `case-*-front.json`: new hash-bound decisions.
- `case-*/canonicalization.json`: outcome, declared calls and supporting evidence.
- Camera `canonical.glb`: rigid export with inverse-transform geometry verification.
- `evaluation.json`, `camera-preview/`, `comparison.jpg`: scoring and visual evidence.

The tracked [results](2026-10-27-photo-evidence-results.json) preserve the protocol,
decisions, comparison and relevant manifest hashes. Dataset-reference angles are
reported separately from the visible evidence; they are audited source-coordinate
references, not independent human semantic labels. Dataset admission remains
ungranted.

The new evidence contract rejects changed images/manifests, wrong mesh/category
bindings and missing image IDs before export. Crop tests cover instance identity,
category mismatch, RGB mask-channel interpretation and absent visible objects.
No new dependency, model checkpoint or backend service was introduced.
