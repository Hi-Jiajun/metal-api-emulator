# B1: synchronous Vulkan provider

`VulkanComputeProvider` implements the core `ComputeProvider` trait for a
single serial exact-thread pass, owned byte views and complete host readback.
The existing `ComputeExecutor` remains available as an independent application
entry point. Both paths share Vulkan execution machinery and the queue lock.

## Invocation and ownership

1. Create a Vulkan executor, then `VulkanComputeProvider::with_executor` to use
   its device. A provider receives a unique process-local device epoch.
2. Call `compile_pipeline(&Function, logical_digest)`. The provider retains the
   translated artifact and returns `CompiledComputePipeline` metadata: epoch,
   pipeline ID, function identity and reflection-derived contract. The logical
   digest is caller-issued fixture metadata, not a content-verified digest or
   artifact-cache key. Pipeline IDs always designate separately registered
   artifacts. Compilation is currently a Vulkan-provider method, not yet a
   cross-provider compilation trait.
3. Build a `ComputeTrace` with that metadata and a `ResourceTableSnapshot`.
   `capabilities().validate_trace(...)` produces the immutable input.
4. `submit` rechecks the receiving device's capabilities and registered artifact,
   including the exact function identity and reflected contract. It refuses
   stale epochs, unknown pipelines, forged reflection, unsupported storage and
   narrowing overflow before creating request-specific Vulkan objects.
5. A successful `submit` returns `CompletedVisible` and canonical writebacks
   sorted by `(allocation_id, view_id)`. Offsets are allocation-relative; only
   writable views are returned, each exactly once at its complete declared size.
6. `wait` observes the recorded terminal disposition. `release_completion` drops
   that observation; `release_pipeline` removes a registry entry. In-progress
   submissions keep their own artifact reference. Records otherwise remain until
   the provider is dropped. These calls do not release abandoned GPU work.

The API supports synchronous commit only. `wait(timeout)` does not launch a
second wait for this implementation; the returned token is already terminal.
There is no end-to-end deadline on compilation, locks, initialization or submit.
The existing 20-second fence timeout makes the executor unusable and reports
unknown completion with retained resources. Live guest leases, multi-pass
ordering, general MTLB resolution and native Metal remain unimplemented.

## Execution failures and visibility

Failures retain Resolve/Compile/Encode/Submit/Wait/Readback phase information.
Vulkan return codes determine device-loss classification; diagnostic strings
are not parsed. This mapping is an experimental Vulkan mapping, not an agreed
native/Vulkan error vocabulary.

- Before queue submission: `NotSubmitted`.
- Queue OOM failures: `NotSubmitted`, because Vulkan guarantees referenced
  resource state is unaffected.
- Other queue failures and fence-wait failures: completion is unknown; handles
  remain retained. Device loss marks the shared executor unusable.
- Readback failure after fence completion: `Failed`, with a token, without
  claiming CPU-visible results.

The queue failure treatment follows the
[vkQueueSubmit failure contract](https://docs.vulkan.org/refpages/latest/refpages/source/vkQueueSubmit.html).
The shared execution path now includes a compute-shader-write to host-read
memory barrier before fence completion, following the
[Khronos host readback example](https://docs.vulkan.org/guide/latest/synchronization_examples.html#_cpu_read_back_of_data_written_by_a_compute_shader).
HOST_COHERENT memory is still required. Unknown terminal results are not proof
that backing storage can be reclaimed; this implementation deliberately keeps
its existing abandonment policy.

## Reproduction

```sh
cargo test --workspace --locked
cargo run --locked -p metal-smoke --bin provider-smoke
```

The provider runner shares one Vulkan device with the old snapshot executor,
compiles each path independently and compares both against the fixture golden.
It covers textual/raw/wrapped copies, the indexed 10x3 boundary/barrier case,
nonzero view offsets, forged pipeline metadata, isolation between two providers
on one executor, unknown completion tokens and use after registry release.
This is Vulkan-provider versus Vulkan-executor regression coverage, not native
Metal parity.

## Verification of this local increment

- 58 unit tests passed: 41 core and 17 Vulkan. New cases include result identity,
  writable-view coverage, premature writeback, range/order validation, bounded
  ID generation, narrowing and execution-failure classification.
- The optional reims integration's 3 adapter tests also passed after the
  shared smoke library change; its 3 pre-existing dead-code warnings remain.
- Formatting, workspace Clippy with `-D warnings` and rustdoc with
  `RUSTDOCFLAGS=-D warnings` passed on Rust 1.96.0.
- Linux/Lavapipe: all four provider comparison cases and rejection/release
  checks passed. The original standalone smoke also passed its four cases.
- Windows GNU debug binaries linked. Running `provider-smoke.exe` directly via
  Windows PowerShell reported NVIDIA GeForce RTX 5060 and passed all four
  comparison cases and rejection/release checks. No VM or deployment-file
  replacement was needed. This does not validate the PowerShell script-loading
  issue recorded for the earlier publication candidate.
- Windows provider executable SHA-256:
  `e8b61a45a64e44850fe048af52420dc2d93fa14feea9d571dd51bf0bf03da661`.
- Device-loss/timeout mappings are unit-tested; no real GPU device loss or
  timeout was injected. No Vulkan validation layer was available locally.
- Initial publication CI at `9d7c007` passed. The workflow now includes
  `provider-smoke`, but this increment has not been pushed or run in remote CI.

No native Metal provider, production reims routing, guest-memory lifecycle or
VM/display result is claimed for this step.
