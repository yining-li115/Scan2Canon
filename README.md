# Scan2Canon

Turn multi-view captures of real-world objects into 3D assets with consistent semantic orientations.

This version provides a local backend scaffold for importing images, submitting jobs, checking inputs, recording events, recovering interrupted jobs, and querying status over HTTP. VGGT, COLMAP, VLMs, and mesh generation are not yet integrated. By default, jobs become `blocked` at the `reconstruct` stage; no placeholder geometry is presented as a real reconstruction.

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

`import` returns a capture ID, and `submit` returns a job ID. Provide 3–200 JPEG or PNG images of the same stationary object in one directory; subdirectories are not scanned. Each image must be no larger than 25 MiB and 50 megapixels, with a maximum of 512 MiB per capture. Three distinct files satisfy the input requirement but do not establish adequate viewpoint coverage. HEIC support, video frame extraction, and mobile uploads are not yet implemented.

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
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
```

Dependency versions are recorded in `uv.lock`. Update the lockfile when changing dependencies. Keep runtime environments and captured data out of version control.

The next step is to implement a real reconstruction adapter and measure runtime, memory usage, and failure rates on a small sample, followed by meshing and canonicalization. Future VLM policies should use the same tools and validation boundaries as the fixed-sequence baseline to support controlled comparisons.
