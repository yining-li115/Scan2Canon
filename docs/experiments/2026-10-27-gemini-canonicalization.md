# Gemini canonicalization: first live diagnostic

The native Gemini worker completed one live mesh-to-canonical run with
`gemini-3.8-flash` on October 27, 2026. This was a new API inference run, separate
from the earlier recorded-decision integration test. The user confirmed the
key's project was on Free Tier with no paid billing enabled. No provider/model
fallback or billing changes were made.

## Input and protocol

- One NOCS mug: `case-06940288c3fa549a`, already used in prior development diagnostics.
- Existing GLB, category hypothesis `mug`, geometry-derived candidates and project
  semantic conventions. No real photographs or prior decisions were supplied.
- Up candidates from surface PCA/support geometry; front candidates generated
  only after saving and validating the up decision.
- Maximum four generation calls, at most two per axis; no automatic HTTP retries.
- Saved decisions were scored afterward against the audited NOCS reference frame.
  Reference coordinates and known input rotations were not model inputs.

This is a single-object diagnostic, not a new holdout accuracy estimate or a
rotation-consistency evaluation.

## Observed result

| Measurement | Result |
| --- | --- |
| Scope | `full_frame` |
| Up / front candidate | 9 / 3 |
| Generation calls | 2: one per axis |
| Extra-view rounds | 0 |
| Human object-direction confirmations | 0 |
| Up error against audited reference | 0.2886 degrees |
| Full-frame error against audited reference | 6.1838 degrees |
| Inverse-transform vertex error | 2.03e-17, below the 2.58e-7 tolerance |
| Export topology | 12,444 vertices; 16,686 faces, preserved |
| Resume | Same result; zero additional model calls |
| Dataset approval | `not_granted` |

The local six-view preview shows an upright mug with its opening upward and its
handle along the project's reference-front direction. The scan's missing bottom
surface remains present; rigid canonicalization does not repair geometry.

The API returned 18,208 input tokens, 267 final-output tokens and 709 thinking
tokens across the two calls (19,184 total). Returned model version was
`gemini-3.8-flash`. Token usage is recorded as usage, not a billing receipt.

## Limits exposed by this run

The upright answer cited only its selected candidate, despite the prompt asking
for an explicit comparison with an alternative. The existing schema permits this;
passing validation does not establish complete compliance with the evidence
protocol. The front answer did compare all four candidates. Preserve this first
result rather than correcting it after reading reference errors.

No ambiguity, supplemental-photo, quota-exhaustion or recovery-after-network-error
branch was exercised by this live run. Tests cover those control-flow behaviors
with fixtures; model behavior on ambiguous objects still requires evaluation.
Agreement with a prior policy on this one mug does not establish general accuracy
or validate all 37 pilot objects.

## Artifacts

Local data, raw replies and renders are retained under
`data/experiments/runs/gemini-free-pilot-01/` and excluded from Git:

- `execution-authorization.json`: user-confirmed tier and call cap.
- `job/request.json`: input, convention, code and policy provenance.
- `job/calls/`: actual requests (image hashes), final replies and token usage.
- `job/decisions/`: separately validated up and front decisions.
- `job/work/export-1/canonicalization.json` and `canonical.glb`: rigid proposal.
- `preview/contact-sheet.jpg`: canonical inspection views.
- `verification.json`: post-inference reference scoring and resume check.

The earlier `preflight.json` is a preserved record of the preparation state before
Free Tier confirmation. Execution authorization and the completed worker state
record the subsequent run. Credentials are never part of the recorded artifacts.
