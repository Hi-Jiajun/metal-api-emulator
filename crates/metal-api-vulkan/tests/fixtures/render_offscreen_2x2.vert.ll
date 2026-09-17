; Owned synthetic AIR counterpart of the reviewed offscreen fixture's vertex
; stage (`conformance/shaders/render_offscreen_2x2.metal`, whose SPIR-V sibling
; is `crates/metal-api-vulkan/src/render_spv/fullscreen_triangle.vert.spvasm`).
;
; The three positions come from `[[vertex_id]]` alone, exactly as the reviewed
; MSL module states them: (-1,-1), (3,-1), (-1,3). The oversize triangle covers
; every pixel centre of a 2x2 viewport, which is what keeps "the draw really
; ran" falsifiable, and the pipeline binds no vertex stream
; (`VertexLayout::None`).
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <4 x float> @render_fullscreen_triangle(i32 %vertex_id) {
entry:
  %second = icmp eq i32 %vertex_id, 1
  %third = icmp eq i32 %vertex_id, 2
  %x = select i1 %second, float 3.000000e+00, float -1.000000e+00
  %y = select i1 %third, float 3.000000e+00, float -1.000000e+00
  %p0 = insertelement <4 x float> undef, float %x, i32 0
  %p1 = insertelement <4 x float> %p0, float %y, i32 1
  %p2 = insertelement <4 x float> %p1, float 0.000000e+00, i32 2
  %p3 = insertelement <4 x float> %p2, float 1.000000e+00, i32 3
  ret <4 x float> %p3
}

!air.vertex = !{!0}
!0 = !{ptr @render_fullscreen_triangle, !1, !3}
!1 = !{!2}
!2 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!3 = !{!4}
!4 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vertex_id"}
