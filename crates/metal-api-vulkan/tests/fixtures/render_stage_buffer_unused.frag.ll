; Owned synthetic AIR for the "declared but never dereferenced" shape (R9c):
; the fragment stage takes a `[[buffer(0)]]` argument and stores a constant
; without ever loading from it.
;
; The metadata says `air.read` (that is what the AIR carries), but the module
; the translator emits never touches the pointer, and the reflection reports
; the refined classification the module itself states: `Unused`. The contract
; admits `BufferAccess::Read` alone, so this stage's declaration disagrees with
; what the module does, and the registration gate refuses the pair by name
; (`render_stage_reflection_mismatch`, field `bindings`) instead of binding
; bytes no declared access covers.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <4 x float> @render_stage_buffer_unused_rgba8(ptr addrspace(1) %tint) {
entry:
  %r = insertelement <4 x float> undef, float 0.250980406999588, i32 0
  %g = insertelement <4 x float> %r, float 0.501960813999176, i32 1
  %b = insertelement <4 x float> %g, float 0.7529411911964417, i32 2
  %a = insertelement <4 x float> %b, float 1.000000e+00, i32 3
  ret <4 x float> %a
}

!air.fragment = !{!0}
!0 = !{ptr @render_stage_buffer_unused_rgba8, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"tint"}
