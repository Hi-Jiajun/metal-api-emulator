# Per-pass resource subsets

V7 allows each dispatch to bind the resources required by its selected compute
pipeline. A buffer may first appear after the first dispatch. The providers
collect and validate the complete trace before encoding, upload every unique
view once, then return one final writeback per ever-writable view.

The trace supports at most eight serial passes and 64 unique views. Rebinding a
view preserves its allocation, offset, length and initial bytes. Access and
Metal slot numbers follow the selected pipeline. First-use slot numbers can
repeat for different resources, so providers identify the upload pool by view
identity and encode each pass's own binding map.

This is predeclared resource use, not allocation or CPU upload during GPU
execution. Writable aliasing, async submission, arbitrary shader admission,
rendering and production guest integration remain outside this subset.

## Fixtures

`suite-v7.json` adds 2/4/8-pass cases. A, B, C and D are separate 120-byte
arrays; bias is four bytes. The two-pass case uses A, bias, B and C. The longer
cases add D. All passes use grid 5x3x2 and cycle local dimensions 4x2x2,
8x4x4 and 1x1x1, exercising exact-thread tail dispatch.

| Pass in cycle | Program | Metal slots mapped to views | Effect |
|---|---|---|---|
| 1 | transform_3d | 0=A, 2=bias, 5=B | A += bias; B = A XOR 0xa5a55a5a |
| 2 | copy_3d | 4=B, 9=C | C = B |
| 3 | remap_3d | 1=bias, 3=C, 7=A | A = (C * 7 + bias) XOR 0x3c3ca5a5 |
| 4 | copy_3d | 4=A, 9=D | D = A |

Integer arithmetic wraps modulo 2^32. The eight-pass case repeats the cycle
without reinitializing any buffer. C and D first appear in different passes
at the same slot 9, which tests resource identity independently of slot labels.
Final writebacks include A/B/C, plus D in the longer cases, even when a view
is absent from the last pass. Bias stays read-only.

The new owned `copy_3d` LLVM/MSL pair uses slots 4 (read) and 9 (write). Its
array proof uses XYZ byte strides 4/20/60, with a 120-byte reach. The capture
runners pin both source hashes and all reviewed program layouts. Native source
admission now accepts six exact MSL fixtures; it is not general MSL reflection.
V1-v6 suite files remain byte-for-byte unchanged. The Rust trace schema stays
2 and fixture/report schemas stay 1, under a distinct `compute-buffer-v7` name.

## Validation

- 119 Rust tests passed: core 64, native 7, Vulkan 34 and capture 14.
- 98 Python tests passed, including independent CPU result calculations,
  missing late writebacks, invalid subset maps and guard/landing checks.
- All three v7 cases passed on Linux/Lavapipe and Windows/RTX 5060. Their
  complete result arrays agree. Real Vulkan reflection verified the new copy
  shader's sparse slots, read/write access and XYZ byte strides.
- All 23 v1-v6 Vulkan cases ran again and still match the archived Swift and
  Rust Metal captures from run 33983234385. Provider smoke also passed.
- Formatting, Clippy, rustdoc, Windows GNU build and macOS ARM64 Rust
  typecheck/Clippy passed. The existing `block` dependency emits a future Rust
  compatibility notice on the macOS target.

Subsequent [CI run 34010989175](https://github.com/Hi-Jiajun/metal-api-emulator/actions/runs/34010989175)
passed at commit `5c10dcd`: Swift and Rust native v1-v7 captures all ran, and
all 26 cases per backend agree with Vulkan. Downloaded reports were checked
against the exact source and suite hashes; v7 also matches local RTX 5060.
The native device was Apple Paravirtual, not a bare-metal Apple GPU.

Swift captures complete GPU allocation bytes. Rust providers expose final view
writebacks landed in host-initialized allocations, so their guard checks do not
observe GPU memory outside the uploaded views. Synthetic Python reports test
the comparator and do not count as Metal execution evidence.

## Run

```sh
cargo run --locked -p metal-smoke --bin provider-capture -- \
  --suite conformance/suite-v7.json --output conformance/captures/vulkan-v7.json
python3 conformance/compare.py --suite conformance/suite-v7.json \
  --check conformance/captures/vulkan-v7.json
# On macOS after building NativeOracle.swift:
python3 conformance/run_native.py --oracle conformance/native-oracle \
  --suite conformance/suite-v7.json --output-dir conformance/captures/native-v7 \
  --require-metal
cargo run --locked -p metal-smoke --bin provider-capture -- \
  --backend native-metal-provider --suite conformance/suite-v7.json \
  --output conformance/captures/rust-metal-v7.json
python3 conformance/compare.py --suite conformance/suite-v7.json \
  --native conformance/captures/native-v7/native-metal.json \
  --vulkan conformance/captures/vulkan-v7.json \
  --metal-provider conformance/captures/rust-metal-v7.json
```
