; Owned synthetic AIR counterpart of a *quad* generated from `[[vertex_id]]`
; alone (2026-09-19, census v45's `vertex_span` bucket).
;
; The milestone's reviewed `vertex_id` module carries three positions and is
; therefore blind to the count a layout-free draw names. This fixture is the
; census's own shape instead: a six-vertex quad whose two triangles share the
; seam x = 0, stated so that the *count* is what decides the covered area.
;
;   id 0 -> (-1,-1)   id 1 -> (0,-1)   id 2 -> (-1,1)     left triangle
;   id 3 -> (0,-1)    id 4 -> (0,1)    id 5 -> (-1,1)     left half, upper
;
; A three-vertex draw therefore rasterizes the lower triangle alone, and the
; six-vertex draw covers the whole left half of the attachment. The seam and
; every hypotenuse pass *between* the 4x4 viewport's pixel centres
; (-0.75, -0.25, 0.25, 0.75), so the covered texels of both counts are the
; fixture's own definition rather than a fill rule's answer — which is what
; makes "the rail issued the count the trace named" falsifiable byte for byte.
;
; Only `icmp eq` and `select` are used, the same op set the reviewed
; counterpart (`render_offscreen_2x2.vert.ll`) is written in.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <4 x float> @render_vertex_id_quad(i32 %vertex_id) {
entry:
  %is_one = icmp eq i32 %vertex_id, 1
  %is_two = icmp eq i32 %vertex_id, 2
  %is_three = icmp eq i32 %vertex_id, 3
  %is_four = icmp eq i32 %vertex_id, 4
  %is_five = icmp eq i32 %vertex_id, 5
  %x_one = select i1 %is_one, float 0.000000e+00, float -1.000000e+00
  %x_three = select i1 %is_three, float 0.000000e+00, float %x_one
  %x = select i1 %is_four, float 0.000000e+00, float %x_three
  %y_two = select i1 %is_two, float 1.000000e+00, float -1.000000e+00
  %y_four = select i1 %is_four, float 1.000000e+00, float %y_two
  %y = select i1 %is_five, float 1.000000e+00, float %y_four
  %p0 = insertelement <4 x float> undef, float %x, i32 0
  %p1 = insertelement <4 x float> %p0, float %y, i32 1
  %p2 = insertelement <4 x float> %p1, float 0.000000e+00, i32 2
  %p3 = insertelement <4 x float> %p2, float 1.000000e+00, i32 3
  ret <4 x float> %p3
}

!air.vertex = !{!0}
!0 = !{ptr @render_vertex_id_quad, !1, !3}
!1 = !{!2}
!2 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!3 = !{!4}
!4 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vertex_id"}
