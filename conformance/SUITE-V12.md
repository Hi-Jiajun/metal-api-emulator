# v12: multi-invocation texture cells over a driver-chosen row pitch

[suite-v12.json](suite-v12.json) reads **every cell** of one 4x4 `R32Uint`
sampled texture, in two dispatch shapes that must agree:

| Case | Entry | Grid | Local | Threadgroups |
|---|---|---|---|---|
| `texture_cell_local_4x4` | `read_texture_2d_cell` | 4x4x1 | 4x4x1 | 1 |
| `texture_cell_local_1x1` | `read_texture_2d_cell` | 4x4x1 | 1x1x1 | 16 |

The fixture is `read_texture_2d_cell`: it reads `texel(x, y)` at the thread's
grid position, adds 100 and stores the result at cell `4*y + x`. The texture
holds 0..15, so every case must land `[100..115]` in the 16-word output buffer.

## What v11 cannot see

The v11 case `sampled_texture_first_texel` dispatches one invocation and reads
texel (0, 0) only; its expectation is 16 zero words. A wrong row stride is
therefore invisible to it — the V = 0 row is the one row a tightly packed upload
still gets right. v12 exists so that a defect in that direction fails a real
capture instead of passing silently.

## The defect this suite pins

A host-visible **linear** image is not required to pack rows tightly. Lavapipe
reports `VkSubresourceLayout.rowPitch` = 64 bytes for a 4x4 `R32Uint` image whose
rows hold 16 bytes (`offset` = 0, `depthPitch` = 0). An upload that writes row
`r` at `r * width * 4` then only agrees with the driver on row 0: rows 1..3 land
in bytes the driver never reads, and `texture.read(x, y)` returns 0 for every
texel with V != 0.

Both dispatch shapes fail identically, and so does a single invocation with
constant coordinates — the row pitch is a provider-side upload question, not a
translated-coordinate one. `provider_suite.rs` carries the same fixture as
`provider_multi_invocation_texture_read`, which runs both shapes on any Vulkan
device, including the RTX 5060.

The fix is in `crates/metal-api-vulkan/src/lib.rs`: the upload asks the driver
for `VkSubresourceLayout` and places each row at `offset + row * rowPitch`
(`depthPitch` is consumed as well, although only D2 depth-1 images are admitted
today). A single-row image keeps the previous tightly packed path, which has no
row distance to get wrong.

## Raw before/after on Lavapipe

Unfixed provider, `cargo run --locked -p metal-smoke --bin provider-capture --
--suite conformance/suite-v12.json`:

```
FAIL: case texture_cell_local_4x4 writeback allocation 800/view 810: first differing byte at offset 20: expected 0x68, got 0x64
texture_cell_local_4x4 [100, 101, 102, 103, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
texture_cell_local_1x1 [100, 101, 102, 103, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
```

Fixed provider, same command:

```
PASS capture: vulkan; compute-buffer-v12; 2 cases; host-visible bytes agreement with suite; allocations=host-writeback-landing
texture_cell_local_4x4 copy_in 2 copy_out 1 [100..115]
texture_cell_local_1x1 copy_in 2 copy_out 1 [100..115]
```

## What the comparator enforces

- The texture section is its own allocation: `initial_hex` must match the
  declared `width * height * 4` extent, and every texture allocation/binding is
  unique inside its case.
- Every writable view must be covered by exactly one expected writeback, so a
  capture that lands fewer cells than the fixture names is refused.
- v12 joins v11 in the provider count contract: a provider capture must report
  `copy_in` and `copy_out`, because the texture upload is one copy-in and the
  output allocation is one copy-in plus one copy-out (`copy_in` = 2,
  `copy_out` = 1). The Swift reference oracle reports bytes without counters and
  stays outside it.

`conformance/test_suite_v12.py` holds the structural checks: the pinned source
digests, the hand-checked `[100..115]` expectation, the `0..15` texture image,
the fact that the v11 case reads only texel (0, 0), the count contract, and a
synthetic capture carrying the collapsed row stride, which the comparator must
refuse. Those tests are comparator checks, not GPU evidence.

## What this suite does not claim

- It is not Metal conformance and not an MSL/AIR equivalence claim. The AIR
  fixture is the reviewed input; the MSL counterpart exists for the source-pair
  identity check and for the native rails.
- It does not admit formats other than `r32_uint`, textures other than D2
  single-sample sampled textures, or a storage texture.
- It does not re-run the RTX 5060 capture. The archived `[0,1,0,1,8,9,8,9,0,...]`
  output is consistent with a 32-byte row pitch there, but that remains an
  inference.
- The macOS rails are extended in code (Swift oracle case identity, Rust native
  provider audit) but were not executed here: no Swift compiler, macOS SDK or
  Apple GPU is available locally.
