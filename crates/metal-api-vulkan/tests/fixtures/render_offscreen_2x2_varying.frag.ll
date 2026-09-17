; Owned synthetic AIR for the varying-linkage counterexample: the same 2x2
; offscreen shape, but the fragment stage consumes a `stage_in` varying.
;
; The fixture's vertex stage produces no user varying at all, so a registration
; pairing the two is a linkage Vulkan would only discover at draw time: the rail
; refuses it at registration (`render_stage_reflection_mismatch`, field
; `varyings`) instead.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <4 x float> @render_varying_rgba8(<2 x float> %uv) {
entry:
  %u = extractelement <2 x float> %uv, i32 0
  %v = extractelement <2 x float> %uv, i32 1
  %r = insertelement <4 x float> undef, float %u, i32 0
  %g = insertelement <4 x float> %r, float %v, i32 1
  %b = insertelement <4 x float> %g, float 0.000000e+00, i32 2
  %a = insertelement <4 x float> %b, float 1.000000e+00, i32 3
  ret <4 x float> %a
}

!air.fragment = !{!0}
!0 = !{ptr @render_varying_rgba8, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.fragment_input", !"generated(uv)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
