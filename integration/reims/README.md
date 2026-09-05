# Optional reims engine comparison

This separate Cargo workspace runs the same fixtures as the standalone smoke
against reims-vgpu's Vulkan compute engine. The root workspace does not resolve
or build this integration. No QEMU build or VM is needed.

Prepare the pinned source once (Python 3 and Git required):

```sh
python3 integration/reims/prepare.py
cargo test --manifest-path integration/reims/Cargo.toml --locked
cargo run --manifest-path integration/reims/Cargo.toml --locked --bin reims-smoke
```

The script fetches upstream commit
`69a57dd69a6958e946c03b73e02db331f330f435`, then checks and applies
`compute-facade.patch`. The patch is the source difference between that upstream
base and local adaptation `3f19c66c7af392d4b588430a07119142c5cea8bd`.
It exposes buffer results, validates exact dispatch and device limits, and adds
an explicit synchronous completion entry. The product's asynchronous entry is
preserved. These changes have not been accepted upstream.

For offline preparation, `--source /path/to/reims-clone` reads the same pinned
commit from an existing clone. It always creates a new checkout here; it does
not edit that source clone. Existing destination directories are refused.
Fetched sources and build products are ignored by Git.

Use `python` on Windows if Python is not installed as `python3`. Vulkan, LLVM
and SPIR-V tool requirements are the same as in the root README. Run the two
executables in separate processes; the reims engine serializes execution and
does not offer an end-to-end timeout for initialization or lock acquisition.

This comparison uses two Vulkan executors. It does not establish native
Metal/Vulkan provider parity or replace the production Metal path.
