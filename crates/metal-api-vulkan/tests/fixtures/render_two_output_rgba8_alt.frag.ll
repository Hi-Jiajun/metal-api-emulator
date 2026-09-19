; The twin of `render_two_output_rgba8.frag.ll`: byte for byte the same module
; except that the *dropped* Location 1 texel is `(0, 1, 0, 1)` instead of
; `(1, 0, 0, 1)`.
;
; The pair is what makes "the extra store is discarded" falsifiable rather than
; assumed: on a single-attachment registration the two modules have to land the
; same four bytes per texel, because the only difference between them is a
; store no attachment is behind. A rail that bound the second location's store
; to the attached one, or summed them, lands two different frames here.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <{ <4 x float>, <4 x float> }> @render_two_output_rgba8_alt() {
entry:
  %r = insertelement <4 x float> undef, float 0.250980406999588, i32 0
  %g = insertelement <4 x float> %r, float 0.501960813999176, i32 1
  %b = insertelement <4 x float> %g, float 0.7529411911964417, i32 2
  %a = insertelement <4 x float> %b, float 1.000000e+00, i32 3
  %d0 = insertelement <4 x float> undef, float 0.000000e+00, i32 0
  %d1 = insertelement <4 x float> %d0, float 1.000000e+00, i32 1
  %d2 = insertelement <4 x float> %d1, float 0.000000e+00, i32 2
  %dropped = insertelement <4 x float> %d2, float 1.000000e+00, i32 3
  %out0 = insertvalue <{ <4 x float>, <4 x float> }> undef, <4 x float> %a, 0
  %out1 = insertvalue <{ <4 x float>, <4 x float> }> %out0, <4 x float> %dropped, 1
  ret <{ <4 x float>, <4 x float> }> %out1
}

!air.fragment = !{!0}
!0 = !{ptr @render_two_output_rgba8_alt, !1, !2}
!1 = !{!3, !4}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!4 = !{!"air.render_target", i32 1, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{}
