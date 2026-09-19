; Owned synthetic AIR for the *per-stage* stage-buffer shape (E-SB2,
; `research/docs/23` §117): the vertex stage of the pair that declares
; thirteen `[[buffer(n)]]` arguments between its two stages — seven here and
; six in `render_stage_buffer_six.frag.ll`.
;
; The seven arguments are all dereferenced: `b0` carries the three `float2`
; clip positions at fixed offsets (a static 24-byte footprint, so the contract
; states no stride), and `b1..b6` are folded into the clip position as whole
; `float4` loads. The fixture's bytes leave the six offsets at zero, so the
; position the module returns is exactly `b0`'s triangle — the oversize one
; that covers every pixel centre of a 2x2 viewport — and a rail that dropped
; one of the six extra descriptors, or bound the wrong slot, is readable two
; ways: a mutated offset past the viewport moves the triangle off it, and the
; fragment half's sum changes with its own payloads.
;
; The stage is translated with the provider's canonical namespace layout
; (`metal_api_vulkan::stage_buffer_namespace_layout`, set 1), so the six
; fragment declarations of the pair stay in the translator's own set 0
; (`research/docs/23` §3.3, E-TX9).
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <4 x float> @render_stage_buffer_seven_positions(ptr addrspace(1) %b0, ptr addrspace(1) %b1, ptr addrspace(1) %b2, ptr addrspace(1) %b3, ptr addrspace(1) %b4, ptr addrspace(1) %b5, ptr addrspace(1) %b6, i32 %vertex_id) {
entry:
  %first = load <2 x float>, ptr addrspace(1) %b0, align 8
  %second_at = getelementptr <2 x float>, ptr addrspace(1) %b0, i64 1
  %second = load <2 x float>, ptr addrspace(1) %second_at, align 8
  %third_at = getelementptr <2 x float>, ptr addrspace(1) %b0, i64 2
  %third = load <2 x float>, ptr addrspace(1) %third_at, align 8
  %is_second = icmp eq i32 %vertex_id, 1
  %is_third = icmp eq i32 %vertex_id, 2
  %chosen012 = select i1 %is_second, <2 x float> %second, <2 x float> %first
  %chosen = select i1 %is_third, <2 x float> %third, <2 x float> %chosen012
  %o1 = load <4 x float>, ptr addrspace(1) %b1, align 16
  %o2 = load <4 x float>, ptr addrspace(1) %b2, align 16
  %o3 = load <4 x float>, ptr addrspace(1) %b3, align 16
  %o4 = load <4 x float>, ptr addrspace(1) %b4, align 16
  %o5 = load <4 x float>, ptr addrspace(1) %b5, align 16
  %o6 = load <4 x float>, ptr addrspace(1) %b6, align 16
  %s1 = fadd <4 x float> %o1, %o2
  %s2 = fadd <4 x float> %s1, %o3
  %s3 = fadd <4 x float> %s2, %o4
  %s4 = fadd <4 x float> %s3, %o5
  %offsets = fadd <4 x float> %s4, %o6
  %offset_xy = shufflevector <4 x float> %offsets, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %sum = fadd <2 x float> %chosen, %offset_xy
  %padded = shufflevector <2 x float> %sum, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %clip = shufflevector <4 x float> %padded, <4 x float> <float poison, float poison, float 0.000000e+00, float 1.000000e+00>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  ret <4 x float> %clip
}

!air.vertex = !{!0}
!0 = !{ptr @render_stage_buffer_seven_positions, !1, !2}
!1 = !{!3}
!3 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!2 = !{!4, !5, !6, !7, !8, !9, !10, !11}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float2*", !"air.arg_name", !"positions"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"offset1"}
!6 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"offset2"}
!7 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"offset3"}
!8 = !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"offset4"}
!9 = !{i32 5, !"air.buffer", !"air.location_index", i32 5, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"offset5"}
!10 = !{i32 6, !"air.buffer", !"air.location_index", i32 6, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"offset6"}
!11 = !{i32 7, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vertex_id"}
