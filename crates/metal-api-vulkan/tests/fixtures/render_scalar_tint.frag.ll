; Owned synthetic fragment stage for the scalar vertex-lane increment
; (2026-09-20, census v46's `vertex_format` bucket): it stores the varying
; [`render_scalar_quad.vert.ll`] forwards, with no arithmetic in between, so
; the attachment's texels are the fetched floats' own 8-bit spellings.
;
; The semantic string matches the vertex fixture's output, and the component
; shape (`float4`) is what the rail's varying-linkage check pairs by location.
source_filename = "render_scalar_tint.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-macosx14.0.0"

define <4 x float> @render_scalar_tint(<4 x float> %tint) {
entry:
  ret <4 x float> %tint
}

!air.fragment = !{!0}
!0 = !{ptr @render_scalar_tint, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.fragment_input", !"generated(2tintDv4_f)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"tint"}
!llvm.module.flags = !{!5, !6}
!llvm.ident = !{!7}
!air.version = !{!8}
!air.language_version = !{!9}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = !{!"Apple metal version 32023.884 (metalfe-32023.884)"}
!8 = !{i32 2, i32 8, i32 0}
!9 = !{!"Metal", i32 4, i32 0, i32 0}
