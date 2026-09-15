# Compute buffer suite v9: command buffer boundaries

`suite-v9.json` reuses the three reviewed `subset_chain_*` fixtures from
[suite v7](RESOURCE-SUBSETS.md) unchanged — same shader sources, buffer pool, dispatch
shapes and expected final bytes — and adds a `command_buffers` field that splits
the dispatch sequence across several command buffers:

| Case | Dispatches | Command buffers | Split |
|---|---|---|---|
| subset_chain_two | 2 | 2 | 1 + 1 |
| subset_chain_four | 4 | 2 | 2 + 2 |
| subset_chain_eight | 8 | 4 | 2 + 2 + 2 + 2 |

Each command buffer commits and completes before the next one is recorded, and
each boundary crosses a real data dependency: the first dispatch of every later
command buffer reads a view that an earlier command buffer wrote. The capture
must therefore re-snapshot landed bytes at the commit/wait boundary instead of
replaying the fixture's initial state, and the reported writebacks are the
merged final per-view landing (a view written by several command buffers
appears once, with the last write winning).

This dimension is deliberately narrow. It does not cover concurrent command
buffers on one queue (the object API's reservation rule is exercised directly by
`provider-smoke`), resource-range hazards, queue priority, or device loss. The
suite extends the schema in one direction only: `command_buffers` is required by
`compute-buffer-v9` and rejected by every earlier suite, and both the Rust
capture and the Swift oracle revalidate the partition before running anything.

## Run

Use the same capture and comparison commands as the [main instructions](README.md),
substituting `conformance/suite-v9.json` and fresh output paths.
