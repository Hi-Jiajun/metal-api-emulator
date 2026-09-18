; Owned synthetic AIR for the *old-ceiling* stage-buffer shape (E-SB1,
; `research/docs/23` §108): the 2x2 offscreen fragment stage reading four
; `[[buffer(n)]]` arguments — the largest list the first increment's
; `MAX_RENDER_STAGE_BUFFERS = 4` admitted, and the shape the widened ceiling
; must leave byte-identical.
;
; Every argument is dereferenced once, by a whole `float4` load, and the four
; loads are summed: the attachment's colour is a function of *all four*
; buffers, so a rail that binds three of the four declarations and drops the
; rest cannot land the colour this module states. The payloads are byte/255
; constants (32 in the ordinary slots, 64 in the last), so the sum — 160/255 —
; sits far from a quantisation tie on every driver (`research/docs/23` §3.5).
;
; The descriptor slot each argument fills is the module's own answer through
; the reflection: the translator's default layout puts `[[buffer(n)]]` at
; `DescriptorSet 0 / Binding n`, and the reviewed arrangement overrides the
; set — the six-slot sibling states the same rule for the widened list.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <4 x float> @render_stage_buffer_four_rgba8(ptr addrspace(1) %b0, ptr addrspace(1) %b1, ptr addrspace(1) %b2, ptr addrspace(1) %b3) {
entry:
  %v0 = load <4 x float>, ptr addrspace(1) %b0, align 16
  %v1 = load <4 x float>, ptr addrspace(1) %b1, align 16
  %v2 = load <4 x float>, ptr addrspace(1) %b2, align 16
  %v3 = load <4 x float>, ptr addrspace(1) %b3, align 16
  %s1 = fadd <4 x float> %v0, %v1
  %s2 = fadd <4 x float> %s1, %v2
  %s3 = fadd <4 x float> %s2, %v3
  ret <4 x float> %s3
}

!air.fragment = !{!0}
!0 = !{ptr @render_stage_buffer_four_rgba8, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4, !5, !6, !7}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b0"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b1"}
!6 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b2"}
!7 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b3"}
