; Owned synthetic AIR for the *widened* stage-buffer shape (E-SB1,
; `research/docs/23` §108): the 2x2 offscreen fragment stage reading six
; `[[buffer(n)]]` arguments — the pipeline-level list census v13's deep tail
; states (`stage_buffer_shape_gt4`), past the first increment's four.
;
; Every argument is dereferenced once, by a whole `float4` load, and the six
; loads are summed: the attachment's colour is a function of *all six* buffers,
; so a rail that binds four of the six declarations and drops the rest cannot
; land the colour this module states. The payloads are byte/255 constants (32
; in the ordinary slots, 64 in one mutant, 255 in the alpha slot), so every sum
; sits far from a quantisation tie on every driver (`research/docs/23` §3.5).
;
; The descriptor slot each argument fills is the module's own answer through
; the reflection: the translator's default layout puts `[[buffer(n)]]` at
; `DescriptorSet 0 / Binding n` (`metal2vulkan::reflect::BUFFER_BINDING_RANGE`),
; so the six declarations land as six descriptors in one set — the arrangement
; whose per-set count the contract's ceiling states.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <4 x float> @render_stage_buffer_six_rgba8(ptr addrspace(1) %b0, ptr addrspace(1) %b1, ptr addrspace(1) %b2, ptr addrspace(1) %b3, ptr addrspace(1) %b4, ptr addrspace(1) %b5) {
entry:
  %v0 = load <4 x float>, ptr addrspace(1) %b0, align 16
  %v1 = load <4 x float>, ptr addrspace(1) %b1, align 16
  %v2 = load <4 x float>, ptr addrspace(1) %b2, align 16
  %v3 = load <4 x float>, ptr addrspace(1) %b3, align 16
  %v4 = load <4 x float>, ptr addrspace(1) %b4, align 16
  %v5 = load <4 x float>, ptr addrspace(1) %b5, align 16
  %s1 = fadd <4 x float> %v0, %v1
  %s2 = fadd <4 x float> %s1, %v2
  %s3 = fadd <4 x float> %s2, %v3
  %s4 = fadd <4 x float> %s3, %v4
  %s5 = fadd <4 x float> %s4, %v5
  ret <4 x float> %s5
}

!air.fragment = !{!0}
!0 = !{ptr @render_stage_buffer_six_rgba8, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4, !5, !6, !7, !8, !9}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b0"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b1"}
!6 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b2"}
!7 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b3"}
!8 = !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b4"}
!9 = !{i32 5, !"air.buffer", !"air.location_index", i32 5, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"b5"}
