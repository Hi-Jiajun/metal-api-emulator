# Sources and attribution

This experimental project uses LGPL-3.0-or-later. See LICENSE for the LGPL
additional permissions and COPYING for the incorporated GPL version 3 text.

The project is independent of Apple and of the reims-vgpu maintainers. Metal
and macOS are Apple trademarks. No guest images or Apple shader binaries are
included.

The indexed-dispatch LLVM fixture
`examples/metal-smoke/shaders/kernel_dispatch_threads_boundary_barrier.ll`
derives from the public fixture of the same name in
[steelbrain/metal2vulkan](https://github.com/steelbrain/metal2vulkan/blob/9e0e99a41dc3cb8bb7e288b531f1698a79fd4b1c/validation/fixtures/public/kernel_dispatch_threads_boundary_barrier.ll) at
`9e0e99a41dc3cb8bb7e288b531f1698a79fd4b1c` (LGPL-3.0-or-later).
The local variant removes an i32-to-i64 extension and uses an i32 GEP index
to keep the optional SPIR-V Int64 capability outside this prototype's subset.
The expected output is the upstream-qualified 120-byte Metal result, recorded
as SHA-256 `36d912d3995f7a5448c6008ef4aef6635a354e0a6a104377ee0a5c23d7b11b99`.
This result is a fixture reference, not a claim that native Metal parity has
been verified for this repository. `kernel_copy_word.ll` is an owned synthetic fixture.

`integration/reims/compute-facade.patch` contains modifications to
[steelbrain/reims-vgpu](https://github.com/steelbrain/reims-vgpu) at
`69a57dd69a6958e946c03b73e02db331f330f435`, also LGPL-3.0-or-later.
The original authors' notices remain in the prepared checkout. The patch is
an unpublished local adapter; it is not an upstream-approved architecture.

The pinned metal2vulkan translator and other Cargo dependencies retain their
own license terms and notices. Cargo.lock records the exact resolved sources.

`conformance/shaders/indexed_boundary.metal` is the source counterpart to the
same public boundary/barrier case, with renamed local variables and comments.
Its original source is
[the pinned public MSL fixture](https://github.com/steelbrain/metal2vulkan/blob/9e0e99a41dc3cb8bb7e288b531f1698a79fd4b1c/validation/fixtures/public/kernel_dispatch_threads_boundary_barrier.metal),
under the same LGPL-3.0-or-later terms. `conformance/shaders/copy_word.metal`
is owned synthetic source paired with the existing copy LLVM fixture.

`conformance/shaders/transform_3d.ll` and `transform_3d.metal` are owned synthetic
source counterparts for bounded 3D read/write and sparse-binding tests. They
contain no extracted AIR or other Apple binary artifacts.

`conformance/shaders/mix_3d.ll` and `mix_3d.metal` are owned synthetic source
counterparts derived from this project's transform_3d fixture. Their distinct
integer operation sequence tests pipeline changes within one command buffer.

`conformance/shaders/remap_3d.ll` and `remap_3d.metal` are owned synthetic source
counterparts for per-pipeline binding-order and access-role tests.
