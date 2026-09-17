; Owned synthetic AIR fixture (E-RS4): the sibling of
; `render_fetch_texture_2d.frag.ll`, differing only in the two texels it reads
; (`research/docs/23` §3.3, v105).
;
; Same stage, same binding, same everything else — the coordinates move from
; (1, 0).x / (0, 1).y to (3, 0).x / (0, 3).y, so against the same
; `(64 * i, 64 * j, 0, 255)` texture the frame moves from `40 40 00 ff` to
; `c0 c0 00 ff`. That is the falsification the fetch face owes: the declaration
; and the module are unchanged, so the bytes can only have followed the
; coordinates.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

source_filename = "render_fetch_texture_2d_far.frag.metal"

define <4 x float> @render_fetch_texture_2d_far(ptr addrspace(1) readonly captures(none) %tex) local_unnamed_addr {
entry:
  %left_pair = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex, <2 x i32> <i32 3, i32 0>, i32 0, i32 0)
  %left_value = extractvalue { <4 x float>, i8 } %left_pair, 0
  %left = extractelement <4 x float> %left_value, i64 0
  %right_pair = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex, <2 x i32> <i32 0, i32 3>, i32 0, i32 0)
  %right_value = extractvalue { <4 x float>, i8 } %right_pair, 0
  %right = extractelement <4 x float> %right_value, i64 1
  %red = insertelement <4 x float> undef, float %left, i32 0
  %green = insertelement <4 x float> %red, float %right, i32 1
  %blue = insertelement <4 x float> %green, float 0.000000000000000e+00, i32 2
  %alpha = insertelement <4 x float> %blue, float 1.000000e+00, i32 3
  ret <4 x float> %alpha
}

declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), <2 x i32>, i32, i32) local_unnamed_addr

!air.fragment = !{!0}
!0 = !{ptr @render_fetch_texture_2d_far, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"tex"}
