# Metal API emulator experiment

This workspace is the source-level Metal API facade discussed for the Windows
vGPU development loop. It does not change QEMU or guest packet routing. A
separate adapter can now run the same application against reims-vgpu's
persistent Vulkan compute engine for an off-VM executor A/B.

The first milestone is intentionally narrow:

```text
Device -> CommandQueue -> CommandBuffer -> ComputeCommandEncoder
       -> Buffer bindings -> dispatchThreads -> commit/wait -> readback
```

The application supplies sanitized Metal AIR/LLVM IR or one binary AIR module
(raw LLVM bitcode or an offset-zero LLVM bitcode wrapper), rather than MSL.
Windows has no Metal shader compiler, so pretending that
`newLibraryWithSource` exists would hide the largest missing component.
`metal2vulkan` translates the AIR to SPIR-V when the compute pipeline state is
created. Ordinary MTLB containers remain an explicit refusal: selecting
`Library.function(name)` from a multi-function metallib requires a real name
table resolver, not a scan for the first embedded wrapper. The standalone Vulkan
backend executes exact `dispatchThreads` regions with a finite 20-second fence
timeout. The reims adapter uses the product engine's explicit synchronous
entry; its fence waits are bounded, but lock acquisition, device creation and
pipeline compilation are not an end-to-end deadline.

The same `metal-smoke` source runs two cases against either executor:

- `copy_word` copies `0x67452301` from buffer slot 0 into a poisoned output.
- The same owned source is assembled at run time and repeated through raw and
  wrapped binary AIR; generated `.air` files remain temporary and uncommitted.
- `indexed_boundary_dispatch` launches a `10x3x1` grid at nominal local size
  `8x2x1`. It exercises a barrier, two-dimensional global-thread indexing and
  all four full/tail regions, then checks all 30 words against the qualified
  Apple Metal output.

The indexed fixture is a semantic derivative of metal2vulkan's public
qualified case: its GEP uses an i32 index so the MVP can keep rejecting the
optional SPIR-V `Int64` capability. Its output remains the same 120-byte Metal
golden (`SHA-256 36d912d3995f7a5448c6008ef4aef6635a354e0a6a104377ee0a5c23d7b11b99`).

## Contract checkpoint

- **Observed:** on Windows/RTX 5060, both the standalone Vulkan executor and
  the reims-vgpu engine pass both smoke cases byte-for-byte without a VM.
- **Contract:** command buffers cannot commit with an open encoder; an encoder
  requires one pipeline and dispatch; every bound buffer range must remain live
  through completion; completion is reported only after Vulkan signals its
  fence and readback is visible. Static and affine `GlobalInvocationId` buffer
  footprints are checked against the exact grid before submission.
- **Unknown:** MSL compilation, Objective-C/Swift binary ABI, textures,
  graphics, presentation, heaps, ICBs, argument buffers, and the alias identity
  of one `Buffer` bound through multiple indices. Data-dependent/unbounded
  indexing and other index domains are also refused rather than submitted with
  an unproven reach.
- **Owner:** `metal-api-core` owns API object and command-buffer state;
  `metal-api-vulkan` owns shared translation/validation plus the standalone
  executor; `metal-api-reims-vulkan` maps that validated request to the reims
  engine. Reims itself owns exact-thread region construction.
- **Test:** both executors must run the same two cases without a VM and return
  identical results; core unit tests reject invalid encoder/commit order.

The milestone does not claim `Metal.framework` compatibility or broad Metal
conformance.

The next contract slice is exposed as `metal_api_core::provider`. It contains
only backend-neutral value types for a validated compute trace, buffer leases
and views, dispatch policy, completion disposition, capabilities, writebacks,
and structured provider errors. The existing `ComputeExecutor` remains a
snapshot compatibility adapter; no production rail consumes the new contract
yet.

## Running the smoke app

The standalone executor needs a Vulkan 1.3 loader/ICD with `maintenance4`.
The reims executor follows the product engine's Vulkan capability floor. Both
need `llvm-as`, `llvm-dis`, and `spirv-val`; put them on `PATH`, or set
`METAL_API_LLVM_AS`, `METAL2VULKAN_LLVM_DIS`, and
`METAL2VULKAN_SPIRV_VAL` to their absolute paths.

```sh
cargo run --release -p metal-smoke -- --executor standalone
cargo run --release -p metal-smoke -- --executor reims
```

The Windows wrapper runs those as separate processes so the two engines never
hold Vulkan devices concurrently:

```powershell
powershell -ExecutionPolicy Bypass -File C:\hackintosh\metal-api-emulator\run-smoke.ps1
```

Each executor prints:

```text
PASS copy_word output=0x67452301
PASS binary_air_copy_word encoding=raw output=0x67452301
PASS binary_air_copy_word encoding=wrapper output=0x67452301
PASS indexed_boundary_dispatch words=30 regions=4
```

The reims adapter currently points at the isolated sibling worktree
`worktrees/reims-metal-api-facade-upstream69`; it is adapted to upstream
`reims-vgpu@69a57dd`. It does not modify the active display branch or select a
different guest backend.
