; Owned synthetic AIR for the NDC-y alignment fixture (`research/docs/23`,
; v40): the strictly asymmetric triangle `(-1,0) (1,0) (-1,1)`.
;
; Metal's clip space is +y up, so this triangle covers the *top* half of the
; attachment under the Metal mapping: two texels on the top row (left of the
; hypotenuse) and six on the second, out of an 8x4 attachment. Vulkan's clip
; space is +y down, so the same module lands the mirrored frame unless the
; translation negates the position's y. Every edge misses every pixel centre
; (the horizontal edge sits between two rows of centres, the vertical edge is
; outside the attachment, and the hypotenuse crosses the first/second row
; boundary at a non-centre x), so the coverage has no driver tie rule in it.
;
; The frame is deliberately not symmetric under the y flip: a mirrored
; readback is a different byte string, which is what gives the fixture teeth.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <4 x float> @render_ndc_y_asymmetric(i32 %vertex_id) {
entry:
  %second = icmp eq i32 %vertex_id, 1
  %third = icmp eq i32 %vertex_id, 2
  %x = select i1 %second, float 1.000000e+00, float -1.000000e+00
  %y = select i1 %third, float 1.000000e+00, float 0.000000e+00
  %p0 = insertelement <4 x float> undef, float %x, i32 0
  %p1 = insertelement <4 x float> %p0, float %y, i32 1
  %p2 = insertelement <4 x float> %p1, float 0.000000e+00, i32 2
  %p3 = insertelement <4 x float> %p2, float 1.000000e+00, i32 3
  ret <4 x float> %p3
}

!air.vertex = !{!0}
!0 = !{ptr @render_ndc_y_asymmetric, !1, !3}
!1 = !{!2}
!2 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!3 = !{!4}
!4 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vertex_id"}
