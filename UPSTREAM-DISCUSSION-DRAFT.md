# Upstream Discussion Draft

Suggested title:

> Proposal: standalone Metal API provider emulator for offline Metal/Vulkan parity

Suggested body:

Hello! I have been exploring the canonical Metal provider direction described
in the reims-vgpu discussion. I prepared a small experimental workspace for
offline iteration before attempting any production rail integration:

> `<repository link>`

This is not a `Metal.framework` ABI implementation, and it is not intended to
replace the current reims Vulkan rail yet. The current prototype covers a
deliberately narrow compute-buffer subset:

- a backend-neutral compute trace and provider contract;
- standalone Vulkan and reims Vulkan adapters;
- raw LLVM bitcode and offset-zero wrapped AIR input;
- exact-thread dispatch and bounded buffer footprint validation;
- resource identity, lease, completion, writeback and structured refusal types;
- offline smoke tests on Linux/Lavapipe and Windows/RTX.

The proposed shape is:

```text
canonical Metal semantic trace
        |                   |
 native Metal provider   Vulkan-backed provider
        |                   |
      macOS              Windows
```

The goal is to run the same small Metal semantic traces against a native Metal
reference and the Windows Vulkan provider, without requiring a VM for every
translation-layer iteration. The existing direct Vulkan rail would remain the
control path until provider parity and canonical-rail integration are proven.

I would appreciate feedback on four points:

1. Does this match the intended canonical Metal rail and Metal-on-Metal
   reference direction?
2. Should the backend-neutral provider contract remain in a standalone
   workspace, or should it eventually move into reims-vgpu?
3. Which native Metal compute seam should be wrapped first for a useful parity
   oracle?
4. Would upstream be interested in collaborating on the first compute-buffer
   parity case before any production integration PR?

The current implementation is intentionally incomplete: it does not claim MSL
compilation, general MTLB function resolution, textures, render, blit,
presentation, ICBs, guest memory landing or full Metal conformance. I am
looking for architectural feedback and a suitable first collaboration scope,
rather than asking for this prototype to be merged as-is.

Related context: `<link to the existing Windows/WHPX issue if useful>`.

## Before publishing

- Replace `<repository link>` after the standalone repository is created.
- Remove the local reims worktree dependency from the public default build.
- Keep the reims adapter as an optional integration or a separately pinned
  compatibility package.
- Add repository metadata, CI and a reproducible Windows build description.
- Recheck the final tree for credentials, VM artifacts and generated shader
  files.
