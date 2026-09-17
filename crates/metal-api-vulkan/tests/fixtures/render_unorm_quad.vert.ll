; Owned synthetic fixture for the normalized vertex-storage increment
; (`research/docs/23` §103, E-VF1). It is not derived from a third-party
; metallib: it is a hand-written AIR module whose `[[stage_in]]` interface is
; the shape the gate-3 census's refusals sit in — a `float2` position beside
; four attributes whose shader-side component shapes are the ones the contract's
; normalized storages declare.
;
; The contract this module registers under (`VertexFormat`, `research/docs/23`
; §103) states one storage per attribute:
;
;   location 0  `float32x2`   position    (8 bytes, offset 0)
;   location 1  `unorm8x4`    colour      (4 bytes, offset 8)
;   location 2  `unorm16x2`   weight      (4 bytes, offset 12)
;   location 3  `unorm8x2`    pair        (2 bytes, offset 16)
;   location 4  `unorm16x4`   packed      (8 bytes, offset 18)
;
; The reflected names are the *shader's* types — `float2`/`float4` — because
; the storage width and its normalization are what the vertex format states,
; not something the AIR member carries: a normalized 8- or 16-bit stream is
; read through exactly the same `float2`/`float4` member its `float32` sibling
; is (`crates/metal-api-vulkan/src/render.rs`,
; `air_type_name_names_vertex_format`). This module is therefore the fixture
; that proves the two halves are separable: the same interface is executed over
; four different storages, and the attachment's bytes are the stored integers'
; own quotients.
;
; The tint it forwards selects one component from each normalized attribute —
; red from the 8-bit four-component storage, green from the 16-bit
; two-component one, blue from the 8-bit two-component one and alpha from the
; four-component 16-bit one — so a rail that dropped, reordered or substituted
; any of the four reads a different texel. Nothing is arithmetic: the forwarded
; components are the fetched values themselves, which keeps the fixture's
; expectation a byte comparison rather than a rounding argument.
;
; The position is the reviewed quad's own `float2` clip-space corner.
source_filename = "render_unorm_quad.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-macosx14.0.0"

define <{ <4 x float>, <4 x float> }> @render_unorm_quad_vertex(<2 x float> %position, <4 x float> %colour, <2 x float> %weight, <2 x float> %pair, <4 x float> %packed) {
entry:
  %extended = shufflevector <2 x float> %position, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %clip = shufflevector <4 x float> %extended, <4 x float> <float poison, float poison, float 0.000000e+00, float 1.000000e+00>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %red = extractelement <4 x float> %colour, i32 0
  %green = extractelement <2 x float> %weight, i32 1
  %blue = extractelement <2 x float> %pair, i32 1
  %alpha = extractelement <4 x float> %packed, i32 3
  %t0 = insertelement <4 x float> poison, float %red, i32 0
  %t1 = insertelement <4 x float> %t0, float %green, i32 1
  %t2 = insertelement <4 x float> %t1, float %blue, i32 2
  %tint = insertelement <4 x float> %t2, float %alpha, i32 3
  %with_position = insertvalue <{ <4 x float>, <4 x float> }> undef, <4 x float> %clip, 0
  %output = insertvalue <{ <4 x float>, <4 x float> }> %with_position, <4 x float> %tint, 1
  ret <{ <4 x float>, <4 x float> }> %output
}

!air.vertex = !{!0}
!0 = !{ptr @render_unorm_quad_vertex, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!3 = !{!"air.vertex_output", !"generated(2tintDv4_f)", !"air.arg_type_name", !"float4", !"air.arg_name", !"tint"}
!4 = !{!5, !6, !7, !8, !9}
!5 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"position"}
!6 = !{i32 1, !"air.vertex_input", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"float4", !"air.arg_name", !"colour"}
!7 = !{i32 2, !"air.vertex_input", !"air.location_index", i32 2, i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"weight"}
!8 = !{i32 3, !"air.vertex_input", !"air.location_index", i32 3, i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"pair"}
!9 = !{i32 4, !"air.vertex_input", !"air.location_index", i32 4, i32 1, !"air.arg_type_name", !"float4", !"air.arg_name", !"packed"}
!llvm.module.flags = !{!10, !11, !12, !13, !14, !15}
!llvm.ident = !{!16}
!air.version = !{!17}
!air.language_version = !{!18}
!air.compile_options = !{!19, !20, !21}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{i32 7, !"frame-pointer", i32 2}
!12 = !{i32 7, !"air.max_device_buffers", i32 31}
!13 = !{i32 7, !"air.max_constant_buffers", i32 31}
!14 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!15 = !{i32 7, !"air.max_textures", i32 128}
!16 = !{!"Apple metal version 32023.884 (metalfe-32023.884)"}
!17 = !{i32 2, i32 8, i32 0}
!18 = !{!"Metal", i32 4, i32 0, i32 0}
!19 = !{!"air.compile.denorms_disable"}
!20 = !{!"air.compile.fast_math_enable"}
!21 = !{!"air.compile.framebuffer_fetch_enable"}
