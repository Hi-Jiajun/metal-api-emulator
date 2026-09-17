; The two-stream fixture beside this file, with the `fast` flag run *removed*
; from its add — the shape a Metal module compiled for `precise` arithmetic (or
; any module whose instruction withholds a floating-point permission) carries.
;
; It exists to witness a boundary rather than to be drawn: the translator
; revision the canonical provider pins decorates a float result with
; `FPFastMathMode` whenever the source instruction withholds a permission, and
; that decoration demands `FloatControls2` + `SPV_KHR_float_controls2`, neither
; of which the provider's Phase-1 capability subset admits. The rail answers
; such a shape with a *typed decline* rather than a silent fallback, which is
; what `a_float_op_that_withholds_a_permission_is_a_typed_decline` beside it
; asserts (the increment's report records the finding and its follow-up).
;
; `reims_indexed_tri_two_stream_precise.air` is `llvm-as` output of this file:
;
;     llvm-as -o tests/fixtures/air/reims_indexed_tri_two_stream_precise.air \
;             tests/fixtures/air/reims_indexed_tri_two_stream_precise.ll
source_filename = "reims_indexed_tri_two_stream_precise.air"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

define <4 x float> @reims_two_stream_vertex(<2 x float> %pos, <2 x float> %offset) local_unnamed_addr {
entry:
  %shifted = fadd <2 x float> %pos, %offset
  %padded = shufflevector <2 x float> %shifted, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %clip = shufflevector <4 x float> %padded, <4 x float> <float poison, float poison, float 0.000000e+00, float 1.000000e+00>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  ret <4 x float> %clip
}

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}
!llvm.ident = !{!7}
!air.version = !{!8}
!air.language_version = !{!9}
!air.compile_options = !{!10, !11, !12}
!air.vertex = !{!13}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 7, !"air.max_device_buffers", i32 31}
!3 = !{i32 7, !"air.max_constant_buffers", i32 31}
!4 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!5 = !{i32 7, !"air.max_textures", i32 128}
!6 = !{i32 7, !"air.max_read_write_textures", i32 8}
!7 = !{!"Apple metal version 32023.884 (metalfe-32023.884)"}
!8 = !{i32 2, i32 8, i32 0}
!9 = !{!"Metal", i32 4, i32 0, i32 0}
!10 = !{!"air.compile.denorms_disable"}
!11 = !{!"air.compile.fast_math_enable"}
!12 = !{!"air.compile.framebuffer_fetch_enable"}
!13 = !{ptr @reims_two_stream_vertex, !14, !15}
!14 = !{!16}
!15 = !{!17, !18}
!16 = !{!"air.position", !"air.arg_type_name", !"float4"}
!17 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"pos"}
!18 = !{i32 1, !"air.vertex_input", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"offset"}
; Copied byte for byte from reims-vgpu `d4ebd12`
; (`crates/reims-vgpu/tests/fixtures/air/reims_indexed_tri_two_stream_precise.ll`,
; sha256 322948f12745269812da65da0a395bc458a28ed027c0e72e39257080ba525e08)
; by the R8 increment of `research/docs/23` §77: the same module as its sibling
; with the `fadd`'s `fast` flag run removed, so the pinned translator decorates
; the result with `FPFastMathMode` and demands `FloatControls2` +
; `SPV_KHR_float_controls2` — the capability the device now answers for.
;
; The two-stream fixture beside this file, with the `fast` flag run *removed*
