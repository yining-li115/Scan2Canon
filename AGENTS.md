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
