# Publication candidate validation

Date: 2026-09-05. Rust: 1.96.0. Scope: standalone-workspace extraction and
reproducible optional reims comparison. This record accompanies the
`publication-prep` branch.

## Checks completed

- Exported the staged Git tree to a fresh directory with no sibling worktrees
  and no prepared reims checkout. Used cached Cargo dependencies and a shared
  build cache; this was not a cold network/dependency-download test.
- In that exported standalone workspace: locked Cargo metadata, formatting,
  44 tests (35 core and 9 Vulkan), warnings-denied workspace Clippy and
  warnings-denied rustdoc passed.
- Exported standalone smoke on Linux/Lavapipe: textual copy, raw AIR, wrapped
  AIR and indexed boundary/barrier dispatch all passed.
- Exported standalone `x86_64-pc-windows-gnu` debug executable linked.
- Optional reims setup: fetched the pinned commit from an existing local Git
  clone into a new ignored checkout, checked and applied the committed patch.
  No edits were made to the source clone.
- Reims adapter: 3 tests, formatting and no-dependency warnings-denied Clippy
  passed. All four shared smoke cases passed on Linux/Lavapipe.
- The integration workspace excludes the prepared upstream workspace; its
  only workspace member is the adapter. Upstream dependencies still emit the
  three previously recorded queue-owner dead-code warnings.
- Current tracked files and 37 historical blobs had no matches for the
  specific GitHub/GitLab-token and private-key patterns checked. This is a
  scoped content check, not a general secret-scanner certification.

The first sandboxed smoke attempts aborted because wait-timeout could not
write to its child-process notification socket (`Operation not permitted`).
The successful runs used the same binaries outside the sandbox and explicitly
selected Lavapipe.

## Not verified in this candidate

- The new Windows wrapper could not load from the WSL UNC share:
  PowerShell reported AuthorizationManager/UnauthorizedAccess, including with
  a per-process execution-policy override. The policy was not changed
  persistently. Native Windows execution of this new candidate is unverified.
- Earlier Windows/RTX 5060 runs cover the preceding two-executor snapshot
  harness; they are historical evidence, not a run of this candidate.
- GitHub Actions has been prepared but not run remotely. The setup script's
  public-network fetch branch was not exercised; the same fixed-commit fetch
  was tested with `--source` pointing at a local clone.
- The manifest's Rust 1.87 minimum has not been tested separately.
- No ComputeProvider implementation, native Metal parity, canonical reims
  routing or VM/display validation is claimed.
