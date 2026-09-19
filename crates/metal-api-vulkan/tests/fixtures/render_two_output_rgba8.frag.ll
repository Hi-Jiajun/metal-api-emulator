; Owned synthetic AIR for the superset fragment interface (2026-09-20, the
; third door behind census v46's `stage_buffer_footprint` bucket).
;
; The module is *unconditionally* two-output, which is the shape the census's
; LPF fragment stage (`fixed_frag_lpf_cpf`, three `Location` stores) has: it
; stores `(64/255, 128/255, 192/255, 1)` at Location 0 — the same texel the
; reviewed offscreen fixture lands, `40 80 c0 ff` in an 8-bit UNORM attachment
; — and `(1, 0, 0, 1)` at Location 1, a texel no single-attachment pass has an
; attachment for. The second store is the reading the increment turns on: when
; only Location 0 is attached, Vulkan discards it, and the frame has to be the
; Location 0 texel byte for byte.
;
; The constants are byte/255 rather than round decimals on purpose: a
; half-integer tie such as `0.5 * 255` is resolved differently by different
; drivers (`research/docs/23` §3.5), while byte/255 values sit far from a tie
; on every driver.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <{ <4 x float>, <4 x float> }> @render_two_output_rgba8() {
entry:
  %r = insertelement <4 x float> undef, float 0.250980406999588, i32 0
  %g = insertelement <4 x float> %r, float 0.501960813999176, i32 1
  %b = insertelement <4 x float> %g, float 0.7529411911964417, i32 2
  %a = insertelement <4 x float> %b, float 1.000000e+00, i32 3
  %d0 = insertelement <4 x float> undef, float 1.000000e+00, i32 0
  %d1 = insertelement <4 x float> %d0, float 0.000000e+00, i32 1
  %d2 = insertelement <4 x float> %d1, float 0.000000e+00, i32 2
  %dropped = insertelement <4 x float> %d2, float 1.000000e+00, i32 3
  %out0 = insertvalue <{ <4 x float>, <4 x float> }> undef, <4 x float> %a, 0
  %out1 = insertvalue <{ <4 x float>, <4 x float> }> %out0, <4 x float> %dropped, 1
  ret <{ <4 x float>, <4 x float> }> %out1
}

!air.fragment = !{!0}
!0 = !{ptr @render_two_output_rgba8, !1, !2}
!1 = !{!3, !4}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!4 = !{!"air.render_target", i32 1, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{}
