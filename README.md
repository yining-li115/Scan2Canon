# Scan2Canon

Turn real-world object captures into generated 3D assets with consistent semantic orientations.

This version provides a local backend scaffold for importing images, submitting jobs, checking inputs, recording events, recovering interrupted jobs, and querying status over HTTP. Independent GPU experiments cover SAM 3D Objects for generative reconstruction and MapAnything as a geometric baseline. Model inference is not yet connected to backend jobs: by default, jobs become `blocked` at the `reconstruct` stage; no placeholder geometry is presented as a real reconstruction.

The current direction is **image + object mask → generated mesh → held-out-view evaluation → semantic canonicalization → review**. Start with [Generative reconstruction](docs/generative-reconstruction.md). A generated asset is a prediction, including unseen surfaces; it must not be labeled as measured geometry or dataset ground truth. The SAM 3D runner records export success separately from validation and dataset approval.

The [semantic orientation experiment](docs/canonicalization.md) renders an existing GLB,
runs pinned Orient Anything V2 inference, and exports ranked pose candidates with
explicit coordinate transforms and disagreement records. Candidates require review;
successful export does not establish a category's semantic front or reconstruction accuracy.

The [semantic front comparison](docs/semantic-orientation.md) adds isolated PointSO
direction prediction and rule-grounded Qwen candidate selection, with shared inputs,
raw evidence, abstentions, and separate accuracy/equivariance evaluation.

For the first GPU experiment, follow [Server experiments](docs/server-experiments.md). The pinned environment and an eight-view inference/export smoke test were verified on an A100 40 GB on October 26, 2026. This confirms execution, not reconstruction accuracy or semantic canonicalization.

## Getting started

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). The local worker uses file locking available on macOS and Linux.

```bash
uv sync --locked
uv run scan2canon import /absolute/path/to/photos
uv run scan2canon submit <capture_id>
uv run scan2canon worker --once
uv run scan2canon status <job_id>
uv run scan2canon events <job_id>
```

`import` returns a capture ID, and `submit` returns a job ID. Provide 3–200 JPEG, PNG, HEIC, or HEIF images of the same stationary object in one directory; subdirectories are not scanned. Each source image must be no larger than 25 MiB and 50 megapixels, with a maximum of 512 MiB of source data per capture. Three distinct normalized images satisfy the input requirement but do not establish adequate viewpoint coverage. Video frame extraction and mobile uploads are not yet implemented.

iPhone HEIC photos can be imported directly using `pillow-heif`; no manual conversion or macOS-specific decoder is required. Import preserves source files byte-for-byte under `originals/` and creates upright, 8-bit RGB JPEGs under `frames/` for model input. Normalization applies orientation, converts embedded ICC profiles to sRGB, and uses JPEG quality 95 without resizing. Unprofiled RGB is assumed to be sRGB; transparent inputs are composited on white. Derived JPEGs omit EXIF/XMP metadata to prevent repeated orientation transforms. Metadata remains in the original files.

The HEIF primary image is used. Auxiliary depth, additional spatial-photo views, Live Photo video, and HDR gain maps are not processed. The JPEG derivative is lossy and does not preserve the original HDR appearance. Keep the original for later processing. Capture manifests record both file hashes and the normalization version; existing version-1 manifests remain readable.

Data is stored in `.scan2canon/` under the current directory by default and is excluded from version control. To use a different location, configure the same data directory for every command and the API:

```bash
export SCAN2CANON_DATA_DIR=/absolute/path/to/scan2canon-data
uv run uvicorn scan2canon.api:create_app --factory --host 127.0.0.1 --port 8000
```

API documentation is available at <http://127.0.0.1:8000/docs>. Starting the API does not execute models; run the worker in a separate terminal. The `worker` command processes queued jobs and exits when the queue is empty. With `--once`, it processes at most one job. It is not a persistent queue daemon.

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{"capture_id":"<capture_id>"}'
```

The HTTP API accepts IDs of previously imported captures, not server filesystem paths. It is intended for trusted local development only. Authentication, user isolation, and upload endpoints are not implemented, so it is not ready for public deployment.

## Project structure

```text
src/scan2canon/
  domain.py          Data contracts, stages, states, and rigid pose validation
  images.py          HEIC/JPEG/PNG decoding and model-input normalization
  ports.py           Interfaces for tools, policies, and storage
  service.py         Capture import and job submission
  policy.py          Reproducible fixed-sequence baseline
  harness.py         Decision execution, budgets, validation, and checkpoints
  runtime.py         Composition root for concrete implementations
  worker.py          Local single-worker locking and interruption recovery
  cli.py             Command-line entrypoint
  api.py             HTTP entrypoint
  storage/           SQLite metadata and local artifacts without overwrites
  tools/             Independent compute adapters
tests/               Recovery, contract, path isolation, and API tests
experiments/mapanything/  Isolated paper baseline, GPU lockfile, and provenance
experiments/sam3d/    Masked image preparation, generative inference, and GPU lockfile
experiments/orientation/  Known-camera rendering, OA V2 inference, and canonical pose candidates
experiments/artifacts.py  Shared checksums and portable experiment manifests
scripts/setup_mapanything.sh  Linux CUDA environment installation and kernel check
scripts/setup_sam3d.sh  Isolated SAM 3D environment and CUDA extension setup
scripts/setup_orientation.sh  Isolated OA V2 GPU environment and source verification
docs/architecture.md Architecture boundaries, limitations, and adapter contracts
```

Keep modules focused instead of introducing generic `utils`, `managers`, or global model singletons. Business logic does not import GPU SDKs; model dependencies and initialization belong in their respective adapters. The scaffold does not currently depend on an agent framework, MCP, Redis, object storage SDKs, or training frameworks.

## Pipeline stages and job states

The stage order is fixed: `inspect → reconstruct → mesh → canonicalize → validate → export`.

Each stage returns named artifact references and numeric metrics. The harness verifies that artifacts exist and belong to the current job. It also checks canonical poses for finite values, orthonormal rotations, and preserved handedness. The `validate` stage must explicitly approve the result before export. These mathematical checks do not establish semantic correctness; semantic quality checks require a separate adapter.

Job states: `queued`, `running`, `blocked`, `needs_review`, `failed`, and `succeeded`.

```bash
# Requeue after resolving the issue; completed stages and attempt counts are retained.
uv run scan2canon retry <job_id>
uv run scan2canon worker --once
```

`retry` does not grant review approval; it reruns the incomplete stage. This version does not support submitting human review evidence, adding reference-library entries, or publishing datasets.

## Development checks

```bash
uv sync --locked --extra canonical
uv run --locked --extra canonical ruff check .
uv run --locked --extra canonical ruff format --check .
uv run --locked --extra canonical mypy src tests experiments
uv run --locked --extra canonical pytest
```

These commands match GitHub Actions and include the CPU orientation geometry/export
tests. The `canonical` extra installs NumPy, trimesh, SciPy and NetworkX, not CUDA. A base-only
installation supports the backend, but not type-checking the optional geometry module.

Backend dependency versions are recorded in `uv.lock`. Each Linux GPU experiment has its own `uv.lock`; it does not install CUDA packages into the backend environment. SAM 3D's compiled extensions use fixed versions/revisions in its setup script, and the installed package inventory is recorded with each run. Update the corresponding lockfile when changing dependencies. Keep runtime environments and captured data out of version control.

The first SAM 3D experiment generated a mesh from one masked video frame on an A100 40 GB: 20.4 seconds of inference and 17.8 GiB peak allocated memory. Other real views remain available for evaluation. MapAnything remains a geometric comparison; its successful smoke test did not establish acceptable object geometry. OA V2 integration has passed local coordinate and export checks; GPU inference and semantic correctness still require verification. Multi-view validation and agent policies remain separate work and are not implied by a successful generation run.
