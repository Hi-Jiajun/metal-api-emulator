; Owned synthetic AIR for the *whole-binding* stage-buffer arm (E-SB3,
; `research/docs/23` §3.3): a fragment stage whose second `[[buffer(1)]]`
; argument is read at an index the translation cannot express.
;
; The first argument is the index itself — one `uint` at the view's first four
; bytes — and the second is a table of `float4` entries the module indexes with
; it:
;
;   %index = load i32,  ptr addrspace(1) %selector
;   %slot  = getelementptr <4 x float>, ptr addrspace(1) %lut, i32 %index
;   %value = load <4 x float>, ptr addrspace(1) %slot
;
; The index is a *value read out of memory*, so the translator's footprint walk
; cannot state where inside `[[buffer(1)]]` the load lands: the reflection
; reports `has_unbounded_access` for that binding and asks every consumer to
; keep the caller's whole window available
; (`metal2vulkan::reflect::BufferFootprint`). `[[buffer(0)]]` keeps its static
; four-byte footprint, so the two declarations of the contract under test state
; the two arms side by side — a static ceiling beside the whole-binding arm.
;
; What makes the frame falsifiable: the selector's own bytes choose the entry,
; so the colour the attachment stores is the *table's* bytes at the *index
; buffer's* value. A rail that bound the wrong window, dropped one of the two
; descriptors, or read the table at a fixed offset lands a different colour.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <4 x float> @render_stage_buffer_range_rgba8(ptr addrspace(1) %selector, ptr addrspace(1) %lut) {
entry:
  %index = load i32, ptr addrspace(1) %selector, align 4
  %slot = getelementptr <4 x float>, ptr addrspace(1) %lut, i32 %index
  %value = load <4 x float>, ptr addrspace(1) %slot, align 16
  ret <4 x float> %value
}

!air.fragment = !{!0}
!0 = !{ptr @render_stage_buffer_range_rgba8, !1, !2}
!1 = !{!4}
!4 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!5, !6}
!5 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"uint*", !"air.arg_name", !"selector"}
!6 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"lut"}
