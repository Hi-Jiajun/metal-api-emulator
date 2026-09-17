; Owned synthetic AIR counterpart of the reviewed offscreen fixture's fragment
; stage (`conformance/shaders/render_offscreen_2x2.metal`, whose SPIR-V sibling
; is `crates/metal-api-vulkan/src/render_spv/solid_unorm8.frag.spvasm`).
;
; The stage stores `(64/255, 128/255, 192/255, 1)`, which an 8-bit UNORM
; attachment reads back as `40 80 c0 ff`. The constants are byte/255 rather than
; round decimals on purpose: a half-integer tie such as `0.5 * 255` is resolved
; differently by different drivers (`research/docs/23` §3.5), while byte/255
; values sit far from a tie on every driver.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <4 x float> @render_solid_rgba8() {
entry:
  %r = insertelement <4 x float> undef, float 0.250980406999588, i32 0
  %g = insertelement <4 x float> %r, float 0.501960813999176, i32 1
  %b = insertelement <4 x float> %g, float 0.7529411911964417, i32 2
  %a = insertelement <4 x float> %b, float 1.000000e+00, i32 3
  ret <4 x float> %a
}

!air.fragment = !{!0}
!0 = !{ptr @render_solid_rgba8, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{}
