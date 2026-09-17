; Owned synthetic AIR for the *affine* stage-buffer shape (R9f): the vertex
; stage reads its clip position out of `positions[vertex_id]`, one `float2`
; per vertex, instead of computing it from `[[vertex_id]]` alone.
;
; The address is `0 + vertex_id * 8` bytes, the exact `constant + stride *
; index` shape the contract's affine footprint states: the reflection reports
; one strided access over `VertexIndex` with stride 8, and the contract
; declares the same access so the pass's view can be proven to cover
; `vertices * 8` bytes before any device object exists.
;
; The fragment half of the pair is the reviewed solid module's synthetic AIR
; counterpart (`render_offscreen_2x2.frag.ll`), so the frame is a function of
; the bytes this stage reads: a rail that dropped the binding or bound the
; wrong element rasterizes a different triangle.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <4 x float> @render_vertex_positions(ptr addrspace(1) %positions, i32 %vertex_id) {
entry:
  %slot = getelementptr <2 x float>, ptr addrspace(1) %positions, i32 %vertex_id
  %xy = load <2 x float>, ptr addrspace(1) %slot, align 8
  %padded = shufflevector <2 x float> %xy, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %clip = shufflevector <4 x float> %padded, <4 x float> <float poison, float poison, float 0.000000e+00, float 1.000000e+00>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  ret <4 x float> %clip
}

!air.vertex = !{!0}
!0 = !{ptr @render_vertex_positions, !1, !2}
!1 = !{!3}
!3 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!2 = !{!4, !5}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float2*", !"air.arg_name", !"positions"}
!5 = !{i32 1, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vertex_id"}
