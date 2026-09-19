; Owned synthetic fixture for the scalar vertex-lane increment (2026-09-20,
; census v46's `vertex_format` bucket). It is not derived from a third-party
; metallib: it is a hand-written AIR module whose `[[stage_in]]` interface is
; the shape the census's remaining refusals sit in — a `float2` position beside
; attributes the shader reads as *scalars*.
;
; The contract this module registers under (`VertexFormat`) states one storage
; per attribute:
;
;   location 0  `float32x2`  position  (8 bytes, offset 0)
;   location 1  `float32x1`  red       (4 bytes, offset 0)
;   location 2  `float32x1`  green     (4 bytes, offset 4)
;
; The two scalar attributes share one stream, so the second one's *offset*
; inside the record is load-bearing: a rail that fetched both storages from
; offset zero would land the first value twice, and both rectangles below would
; collapse onto one texel. The reflected names are the shader's own scalar
; spelling (`air.arg_type_name` `float`), which is what the rail's
; `air_type_name_names_vertex_format` pairs with the one-component storage: a
; vector member declared over it, and this scalar member declared over a vector
; storage, are the two directions of the same refusal.
;
; The tint it forwards is the two scalars themselves — red from the first
; attribute, green from the second, blue zero and alpha the sentinel — with no
; arithmetic in between, so the attachment's bytes are the fetched floats' own
; 8-bit spellings: `0.2` lands `0x33` and `0.6` lands `0x99` under both the
; round-to-nearest and the truncating reading of the float-to-unorm
; conversion, because `0.2f * 255` is `51.000001` and `0.6f * 255` is
; `153.000006`. The two rectangles swap those two values, so the frame is
; `33 99` on one half and `99 33` on the other.
;
; The position is the reviewed quad's own `float2` clip-space corner.
source_filename = "render_scalar_quad.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-macosx14.0.0"

define <{ <4 x float>, <4 x float> }> @render_scalar_quad_vertex(<2 x float> %position, float %red, float %green) {
entry:
  %extended = shufflevector <2 x float> %position, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %clip = shufflevector <4 x float> %extended, <4 x float> <float poison, float poison, float 0.000000e+00, float 1.000000e+00>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %t0 = insertelement <4 x float> poison, float %red, i32 0
  %t1 = insertelement <4 x float> %t0, float %green, i32 1
  %t2 = insertelement <4 x float> %t1, float 0.000000e+00, i32 2
  %tint = insertelement <4 x float> %t2, float 1.000000e+00, i32 3
  %with_position = insertvalue <{ <4 x float>, <4 x float> }> undef, <4 x float> %clip, 0
  %output = insertvalue <{ <4 x float>, <4 x float> }> %with_position, <4 x float> %tint, 1
  ret <{ <4 x float>, <4 x float> }> %output
}

!air.vertex = !{!0}
!0 = !{ptr @render_scalar_quad_vertex, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!3 = !{!"air.vertex_output", !"generated(2tintDv4_f)", !"air.arg_type_name", !"float4", !"air.arg_name", !"tint"}
!4 = !{!5, !6, !7}
!5 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"position"}
!6 = !{i32 1, !"air.vertex_input", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"red"}
!7 = !{i32 2, !"air.vertex_input", !"air.location_index", i32 2, i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"green"}
!llvm.module.flags = !{!8, !9, !10, !11, !12}
!llvm.ident = !{!13}
!air.version = !{!14}
!air.language_version = !{!15}
!air.compile_options = !{!16, !17, !18}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 7, !"frame-pointer", i32 2}
!10 = !{i32 7, !"air.max_device_buffers", i32 31}
!11 = !{i32 7, !"air.max_constant_buffers", i32 31}
!12 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!13 = !{!"Apple metal version 32023.884 (metalfe-32023.884)"}
!14 = !{i32 2, i32 8, i32 0}
!15 = !{!"Metal", i32 4, i32 0, i32 0}
!16 = !{!"air.compile.denorms_disable"}
!17 = !{!"air.compile.fast_math_enable"}
!18 = !{!"air.compile.framebuffer_fetch_enable"}
