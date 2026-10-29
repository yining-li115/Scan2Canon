# Engineering conventions

- Keep domain contracts independent of web frameworks, storage engines and model SDKs.
- Select concrete implementations in `runtime.py`; avoid hidden initialization on import.
- Add an abstraction only for a real boundary or a demonstrated second implementation.
- Keep CLI and HTTP routes thin. Long compute belongs in tools executed by workers.
- Preserve raw captures and completed-stage artifacts. Record tool and convention versions.
- Never fabricate geometry or report missing adapters as successful work.
- Do not let policy output bypass validation, execution budgets or review requirements.
- Keep GPU dependencies optional and isolated to their adapters.
- Do not include generator signatures, attribution footers or co-author trailers in code or commits.
- Use `uv.lock` and run Ruff, mypy and relevant pytest tests before finishing changes.
- Test failure recovery and externally visible behavior, not private implementation details.
- Do not add new services, generic workflow frameworks or large dependencies without a concrete need.

# Pose visualization

- Every user-facing pose result or comparison must overlay labeled X/Y/Z axes using the actual pose and render camera. Use red for X, green for Y, and blue for Z; the canonical basis is right-handed, +Y up and +Z reference front when established.
- Clearly distinguish proposed or unresolved directions from confirmed ones. An up-only export must not visually imply that its front is established.
- Keep inference evidence unchanged; create axis overlays as separate presentation artifacts so annotations do not leak into model inputs or overwrite raw renders.
