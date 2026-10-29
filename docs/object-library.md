# Object library

The local library persists **instance semantic proposals and their evidence**.
It does not classify objects automatically, establish category standards or certify
canonical poses. Inference still runs in the existing evidence-agent worker; this
library is a separate persistence boundary, not another agent or workflow engine.

## Current implementation and deployment target

The current adapter is SQLite, composed in `src/scan2canon/runtime.py` and stored
in `<data-dir>/library.sqlite3`. Existing jobs remain in `jobs.sqlite3`; capture
and experiment files are unchanged. No new dependencies, GPU requirements,
services, model calls or credentials are required.

The planned shared-service target is PostgreSQL, adding pgvector only when an
embedding-based retrieval experiment is actually implemented. Relationships and
stable identifiers will live in relational columns, evolving claim payloads in
JSONB, and large files in object storage. A vector index retrieves candidates;
it does not establish category membership or equivalent canonical directions.
No PostgreSQL adapter, embedding model, vector index, object-storage service or
migration command is implemented yet. Migration will require explicit schema and
artifact-location conversion, not just changing a connection string.

## Contracts

`src/scan2canon/library.py` is independent of database engines, HTTP and model SDKs.
`ObjectLibrary` in `ports.py` defines the persistence boundary. `SQLiteLibrary`
implements it. The experimental import adapter translates canonicalization output
without introducing experiment imports into the application domain.

| Record | Meaning | Important boundary |
| --- | --- | --- |
| `ObjectRecord` | Stable instance ID, display name, source-mesh hash | Display names are not category labels; an instance is a reconstruction, not proven physical-object identity |
| `Assessment` | Functional identity alternatives, appearance, parts, direction meanings, selected axes and model provenance | Always `proposed`; the model's `semantic_status=resolved` is its claim, not independent verification |
| `EvidenceRef` | Evidence ID, location, hash and kind | A hash establishes file integrity, not truth; evidence is referenced, not regenerated |
| `MatchProposal` | Relation between two exact assessment versions plus optional up/front correspondences | Identity and axis compatibility are separate; every match remains a proposal |

Direction claims contain their role, description, observable `from_part → to_part`
and image citations. Ambiguities, numeric measurements, checks, transformations,
raw replies and all source versions remain in the referenced original reports and
request artifacts. In particular, an up-only result never establishes front.
No category-name menu, manually authored direction table or convention lookup is added.

A match can propose shared up while leaving front unspecified. The repository
requires two different object instances, existing assessment IDs and matching-axis
direction IDs. It does not pretend these reference checks establish semantic truth.
The [pair comparison worker](shared-frames.md) can now infer a match and a shared
frame proposal from two explicitly selected assessments. It does not search the
library, establish category membership or promote a proposal to a standard.

## Import existing results

From the repository root, after the normal locked environment setup:

```bash
uv run --locked --extra canonical python -m experiments.library_import \
  data/experiments/runs/claude-direction-evidence-02/blue-mug/job \
  --name blue-mug
```

`--name` is only a human-readable display name. `--data-dir /absolute/path/store`
selects a different local catalog. By default the importer derives a deterministic
instance UUID from the **source mesh hash**, and an assessment UUID from that
instance UUID and the final report hash. Re-importing the same job and name is
idempotent. Changed names or conflicting records fail instead of overwriting history.
Different reports can coexist for one unchanged source mesh. Reconstructed meshes
with different hashes stay separate until association is explicitly addressed;
this is not physical-object deduplication. `--object-id UUID` supports an explicit
association to a compatible source-mesh record, never a guess based on appearance.

The importer accepts completed `oa-evidence-agent-v3` jobs ending in `agent_finished`.
Pending/failed jobs stay in the experiment archive and are not imported. It verifies
completed-stage file hashes and safe paths, request and final-call hashes, source
mesh linkage, and equality between the final report and the saved typed model reply.
Only then does one database transaction store the object and assessment. Source
files are never edited. Missing or modified evidence aborts before insertion.

The local catalog stores absolute paths to existing experiment files, not copies.
Keep these directories alongside the database when backing up; a database-only
backup does not preserve images or meshes. Integrity is checked at import time.
Any future agent retrieval tool must recheck hashes before using current file bytes;
plain metadata queries do not revalidate every file. The source report links the
exported canonical GLB and records its transform and hash.

## Query without a model

```bash
uv run --locked scan2canon library list
uv run --locked scan2canon library show OBJECT_UUID
uv run --locked scan2canon library assessments OBJECT_UUID
uv run --locked scan2canon library matches OBJECT_UUID
```

Use `scan2canon --data-dir /absolute/path/store library ...` to query the same store
as an import with `--data-dir`. Listing is paginated with `--limit` (1–1000) and
`--offset`. A missing object raises an error rather than returning a fabricated
empty record. No HTTP endpoints or arbitrary agent SQL tool are exposed.

## Integrity and recovery

Objects, assessments and match proposals are immutable. Repeated identical writes
succeed without duplicates; conflicting writes fail. A failed transaction rolls
back both object and assessment insertion. Foreign keys protect relationship
references. Repository writes revalidate contracts even when a caller bypasses
normal model construction. Schema version 1 is explicit; an unknown database
version fails without downgrading it. Future schema upgrades need explicit migrations.

These are local trusted-development capabilities, not a multi-tenant deployment.
Authentication, ownership isolation, a semantic concept hierarchy, concept merges,
approval/retraction lifecycle and vector retrieval remain future work. Before
connecting the library to agent actions, add evidence-verifying read tools and
validated proposal-write tools; the policy must not directly execute SQL or mark
its own conclusions as established knowledge.

## Two-mug persistence check

On 2026-10-29 the two completed Claude jobs were imported into the local default
catalog, without additional inference:

| Instance | Object UUID | Stored assessment |
| --- | --- | --- |
| Blue mug | `a8021c51-33ce-5775-b11a-98e6f952564a` | `64ea7d87-6c21-5ec1-a23b-8931267c2b5a` |
| Red-handled mug | `3739bac7-1beb-5b43-a965-9f12566f1d5b` | `accd7321-133e-5e88-b740-80a6e4a97bc9` |

Both imported assessments remain up-only proposals. The initial persistence check
inserted no match or category membership by hand. A later model-driven pair
experiment is documented in [shared frames](shared-frames.md). The import itself
verifies persistence/provenance, not learned classification or
cross-instance canonicalization. Tests use clearly synthetic fixtures for match
reference validation, transactional rollback, idempotent recovery, immutable
records, corrupt-artifact rejection and pending-job rejection.
