; Owned synthetic AIR for the translated-stage counterexample: the same 2x2
; offscreen shape, but the fragment stage reads a Metal buffer.
;
; The rail's render stages bind no descriptor set (the contract's vertex layout
; is the whole input side), so a translation that names a buffer is interface
; this rail does not execute yet: the registration refuses it by name
; (`render_stage_unsupported_interface`) instead of executing the stage with the
; binding silently dropped.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <4 x float> @render_buffered_rgba8(ptr addrspace(1) %tint) {
entry:
  %word = load i32, ptr addrspace(1) %tint, align 4
  %red = bitcast i32 %word to float
  %r = insertelement <4 x float> undef, float %red, i32 0
  %g = insertelement <4 x float> %r, float 0.000000e+00, i32 1
  %b = insertelement <4 x float> %g, float 0.000000e+00, i32 2
  %a = insertelement <4 x float> %b, float 1.000000e+00, i32 3
  ret <4 x float> %a
}

!air.fragment = !{!0}
!0 = !{ptr @render_buffered_rgba8, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float*", !"air.arg_name", !"tint"}
