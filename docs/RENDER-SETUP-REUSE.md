# The render-setup reuse

`docs/SUBMIT-PHASE-PROFILE.md` prices one offscreen pass's pre-recording work
separately (`render_setup`). The fp9b round measured it at **1 135.9 µs per
submit — 14.8% of `provider.submit`** — and this increment is the answer to the
part of it that a *shape* decides: the shader modules, the pipeline layout and
the graphics pipeline.

## What a pass builds, and what a shape decides

Every offscreen pass creates its device objects from the request and destroys
them with the pass. Of those objects, three families do not depend on the pass's
*content* at all — only on the shape the pass states:

| object | what the shape is |
|---|---|
| `VkShaderModule` (vertex, fragment) | the SPIR-V words the rail was handed |
| `VkPipelineLayout` | the ordered list of descriptor-set layouts the pass's sets were created from |
| `VkGraphicsPipeline` | the modules and entry names, the vertex input, the topology, the raster/multisample/blend state, the depth-stencil state, the specialization constants, the render pass and that layout list |

Everything else stays per pass, exactly as it was: images, memory, image views,
samplers, descriptor pools and sets, render passes, framebuffers, readback
buffers, command pools, command buffers and fences. The increment moves no
bytes: it hands a pass the same immutable objects a previous pass of the same
shape built, and the pass records, submits and reads back through the paths it
always used.

## The key cannot lie

The key is assembled **from the structures that are about to be handed to the
driver**, not from a second reading of the request:

* the render pass half is read back out of the `VkAttachmentDescription2`,
  subpass references, resolve chain and `VkSubpassDependency2` list that
  `vkCreateRenderPass2` is called with;
* the descriptor-set layouts are read back out of the
  `VkDescriptorSetLayoutBinding` arrays the layouts were created from;
* the pipeline half is read back out of the `VkGraphicsPipelineCreateInfo` and
  its states, with the shader modules as their own words and the layouts as
  their own definitions.

Two passes therefore meet in the cache exactly when the driver would be told
the same thing twice, and the comparison that decides is a full field-by-field
equality — the 64-bit digest is only the bucket. `VkDescriptorSetLayout`
compatibility is defined on the bindings a layout was created with and
`VkRenderPass` compatibility on the attachment and subpass descriptions, so the
reused pipeline is legal beside the pass's own fresh descriptor sets and its
own render pass: the key compared those definitions.

The objects are **taken out** of the cache by the pass that uses them and put
back after the pass's fence has signalled, so the cache never holds an object a
pass is recording with. A pass that fails in between destroys what it took —
the fail-closed direction — and the next pass of that shape builds again.

## The switch, and what it is for

| value | effect |
|---|---|
| unset (default), `1`, `on`, `yes`, `true` | on |
| `0`, `off`, `no`, `false` | off: no lookup, no insert, one relaxed load per pass |

The control arm of a round is the same executable with
`METAL_API_VULKAN_RENDER_SETUP_CACHE=0`, which is what makes the two arms differ
in this mechanism and nothing else. The provider also states the switch directly
(`set_render_setup_reuse`), so both arms can run against one device in one
process — which is how `tests/render_setup_reuse_e2e.rs` compares them.

## Invalidation

The entries are built from contract-surface objects, so they are dropped when
that surface moves:

* **a released render pipeline registration** (`release_render_pipeline`) empties
  the cache: the registration is part of what the entries were minted under, and
  a pass after the release rebuilds exactly as it did before this increment;
* **a device-loss rebuild** installs a whole new context (`rebuild_after_device_loss`
  builds a fresh executor), so the old device's entries go with it;
* **switching the mechanism off** drops what it held;
* **the cap** (128 shapes) evicts the oldest entry, destroying its objects under
  the lock that removed them. The guest desktop repeats a handful of shapes, so
  the cap is a bound on a pathological stream rather than a working limit, and
  `evictions` says when it was reached.

## Reading it

Three surfaces, in the order a round reads them:

1. the profile line's `reuse_*` counters (`docs/SUBMIT-PHASE-PROFILE.md`), which
   partition every pass into served, built, refused by comparison, unkeyable and
   switched off;
2. `VulkanComputeProvider::render_setup_reuse_counts()`, the same counters plus
   the resident entry count and the evictions, readable without the profile;
3. the `setup_*` bars, which say what the assembly costs on the arm that does
   not reuse — `setup_pipeline_us` is the bar this increment moves, and the
   other nine are the ones it deliberately leaves alone.

## Boundaries

The reuse is a timing change inside one provider call. It does not widen the
admitted class, does not change any frame byte, and does not make the provider's
submission cheaper in GPU work: the pass records and waits exactly as before.
The shapes a round's guest actually draws decide how much of `render_setup` the
`setup_pipeline_us` bar can pay for, and a round whose passes all differ in
shape sees `reuse_hit_n=0` with `reuse_miss_n` equal to the pass count — a
reading, not a failure. A reading from this increment is a reading of the
canonical provider on the machine that produced it, and is not complete Metal
conformance evidence.
