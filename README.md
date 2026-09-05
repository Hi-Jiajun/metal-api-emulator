# Metal API emulator experiment

Experimental source-level Metal compute objects backed by Vulkan, for fast
host-side iteration without booting a VM. The long-term proposal is to share
one Metal semantic path between native Metal and a Windows Vulkan provider.
This repository is an independent prototype; upstream has not adopted it.

The working application path is:

```text
Device -> Library/Function -> ComputePipelineState
       -> CommandQueue -> CommandBuffer -> ComputeCommandEncoder
       -> Buffer bindings -> dispatchThreads -> commit/wait -> readback
```

`metal-api-core` owns these Rust objects and their command state machine.
`metal-api-vulkan` translates AIR through the pinned metal2vulkan revision and
executes the buffer-compute subset. An [optional reims integration](integration/reims/README.md)
runs the same fixtures against the reims Vulkan engine in a separate workspace.

## Current status

- Working: synchronous buffer compute, exact-thread dispatch (including tail
  regions), textual LLVM IR, raw AIR bitcode and offset-zero bitcode wrappers.
- Validation: API ordering, foreign pipeline rejection, bounded buffer access,
  device limits and CPU-visible readback. Multiple bindings of the same Buffer
  are detected and refused.
- Experimental: `metal_api_core::provider` contains trace/resource values,
  capability admission and a `ComputeProvider` trait declaration. **No backend
  implements that trait yet.** Real GPU smoke still uses `ComputeExecutor`.
- Open design work: compile/pipeline ownership, asynchronous result retrieval,
  completion-driven resource retention and validation of provider writebacks
  against the admitted trace. Resource snapshots do not hold live guest pages.
- Not implemented: native Metal provider, general MTLB function-name resolution,
  MSL compilation, textures, rendering, presentation, heaps, ICBs or production
  reims integration. This is not a Metal.framework ABI implementation.

The goal is the host provider used by reims and source-level test programs;
loading arbitrary macOS Objective-C/Swift binaries on Windows is outside this
project's current scope. See the [collaboration draft](UPSTREAM-DISCUSSION-DRAFT.md).

## Build and test the standalone workspace

Install Rust, a C linker and Git. The current preparation is tested with Rust
1.96.0; the manifests retain the previous 1.87 minimum, which has not yet been
separately verified. A first build downloads Cargo dependencies, including
metal2vulkan at `9e0e99a41dc3cb8bb7e288b531f1698a79fd4b1c`.

```sh
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked --no-deps -- -D warnings
cargo fmt --all -- --check
```

These commands require no sibling worktrees, reims checkout, GPU or VM.

For live smoke, install a Vulkan 1.3 loader/ICD with maintenance4, `llvm-as`,
`llvm-dis` and `spirv-val`. Put the tools on PATH or set `METAL_API_LLVM_AS`,
`METAL2VULKAN_LLVM_DIS` and `METAL2VULKAN_SPIRV_VAL` to their executable paths.

```sh
cargo run --locked -p metal-smoke -- --executor standalone
```

On Linux, select Lavapipe explicitly when needed by setting `VK_ICD_FILENAMES`
to the installed `lvp_icd*.json` file under `/usr/share/vulkan/icd.d/`.

On Windows, use the Rust GNU target with MSYS2 MinGW-w64 GCC, LLVM and
SPIRV-Tools available on PATH, plus the GPU vendor's Vulkan driver:

```powershell
cargo build --locked --release --target x86_64-pc-windows-gnu -p metal-smoke
.\run-smoke.ps1 -Runner .\target\x86_64-pc-windows-gnu\release\metal-smoke.exe
```

The PowerShell wrapper accepts tool paths from the environment above and also
recognizes the conventional `C:\msys64\mingw64\bin` installation. To run the
optional engine comparison after building it, pass `-ReimsRunner` with the path
to `reims-smoke.exe`. Both executables run in separate processes.

The suite checks:

```text
PASS copy_word output=0x67452301
PASS binary_air_copy_word encoding=raw output=0x67452301
PASS binary_air_copy_word encoding=wrapper output=0x67452301
PASS indexed_boundary_dispatch words=30 regions=4
```

The indexed case launches a 10x3 grid with an 8x2 nominal threadgroup, exercises
a barrier and checks all 30 output words. Its source and reference output are
attributed in [NOTICE.md](NOTICE.md). Generated AIR stays temporary.

## Evidence and limits

Earlier local checkpoints ran these four checks on Linux/Lavapipe and
Windows/RTX 5060 with both Vulkan executors. Those runs validate the narrow
snapshot executor path. They do not demonstrate native Metal parity or the
new provider trait. The current publication-preparation checks are recorded in
[docs/VALIDATION.md](docs/VALIDATION.md).

The standalone fence wait has a 20-second bound. Reims uses its explicit
synchronous retirement entry. Neither interface provides an end-to-end
initialization/compilation/lock deadline. Guest memory, display and VM behavior
remain outside this offline test boundary.

## License

LGPL-3.0-or-later; see [LICENSE](LICENSE), [COPYING](COPYING) and
[NOTICE.md](NOTICE.md) for the source attribution and license texts.
