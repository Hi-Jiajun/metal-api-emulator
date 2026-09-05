# Proposal: offline Metal API provider prototype and first parity case

Hello! Following steel-brain's Discord suggestion to use a shared Metal
semantic path with a Windows Metal API emulator, I have been prototyping an
independent host-side compute workspace:

Proposed repository: https://github.com/Hi-Jiajun/metal-api-emulator

The working subset is Rust Device/Queue/CommandBuffer/ComputeEncoder/Buffer
objects backed by Vulkan, accepting textual LLVM IR and single-module binary
AIR. The standalone smoke runs without reims or a VM. A separate optional
workspace exercises the same fixtures through a pinned reims Vulkan engine.
Earlier local runs passed on Linux/Lavapipe and Windows/RTX 5060, covering a
buffer copy and a 10x3 indexed dispatch with a barrier and exact tail regions.

There is also a draft backend-neutral trace/resource model and capability
validation. `ComputeProvider` is currently an interface declaration with no
backend implementation. Asynchronous readback, native Metal execution and
production integration are still open work. The existing two-executor
comparison is Vulkan versus Vulkan, not native Metal/Vulkan parity.

I would like to use this as a small collaboration experiment toward:

```text
shared host Metal operations
  -> native Metal provider on macOS
  -> Vulkan-backed provider on Windows
```

The first milestone I propose is one shared buffer-compute case, with native
Metal output/completion as the reference and Windows Vulkan as the target.
I can contribute the Windows/RTX and WHPX test environment; help with the
native Metal reference and the intended API boundary would be especially useful.

Before expanding the interface, could we align on:

1. Whether this trace/provider approach fits the intended Metal API emulator,
   or whether matching existing Metal call sites more closely is preferable?
2. Whether the contract should remain independent or eventually live in reims?
3. Which native compute call site and one small parity case would make the
   most useful first collaboration milestone?

This does not implement Metal.framework binary compatibility, MSL compilation,
textures/render/presentation or guest-memory lifecycle. It has not replaced a
production backend or resolved the VM display bugs. I am seeking feedback on
the boundary and first experiment, with focused integration PRs following once
we have evidence.
