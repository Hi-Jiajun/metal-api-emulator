; Owned synthetic AIR fixture (E-TC1): a fragment stage that declares
; *thirteen* sampled `texture2d<float, sample>` arguments, one at each Metal
; index 0 through 12, and folds every one of them into the colour it returns.
;
; The shape is the census's own: the preview round behind this increment read
; 42 class exits under `render_provider_out_of_class_texture_count`, and every
; one of them was a fragment stage declaring thirteen sampled textures while
; the canonical contract stated eight for the pass's whole list
; (`/mnt/c/tmp/reims-vgpu-fail.log`). The contract now states one *stage's*
; window instead, so the module below is the shape those rows were asking for.
;
; Every texture is a 4x4 `rgba8_unorm` surface whose texel `(1, 0)` holds
; `k + 1` in red (texture `k`), zero in green and blue and full alpha, and the
; body samples each one at that texel's centre — a linear filter's weight is
; one there, so the readings are the uploaded bytes rather than a driver's
; filtering precision. The returned colour is
;
;   red   = sum of all thirteen readings  (1 + 2 + ... + 13 = 91 -> 0x5b)
;   green = texture 0's reading           (1 -> 0x01)
;   blue  = texture 12's reading          (13 -> 0x0d)
;   alpha = 1.0                           (0xff)
;
; so every one of the thirteen textures' own bytes moves the frame: dropping
; any single contribution moves red by at least one unorm step, and the two
; channels beside it pin the first and the last argument of the table. The
; state is the module's own Nearest + ClampToEdge sampler, which the contract's
; declarations repeat (`research/docs/23` §3.3, v100).
;
; Each argument carries its *own* AIR sampler-state global (thirteen identical
; Nearest + ClampToEdge words under the frontend's suffix numbering) because
; the rail pairs a module's static samplers with its texture declarations
; positionally: one sampled texture is read through one state, so the shape a
; stage with thirteen arguments states is thirteen states beside them.
;
; The adds carry the `fast` flag run because that is what the frontend emits
; for the module's own arithmetic: an AIR float op that *withholds* a
; floating-point permission is decorated with `FPFastMathMode` by the pinned
; translator and demands `FloatControls2` + `SPV_KHR_float_controls2`, which
; the provider's Phase-1 capability subset does not admit
; (`tests/render_translated_stage.rs::a_float_op_that_withholds_a_permission_is_a_typed_decline`).
; A module written the way the frontend writes it stays inside the subset.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

source_filename = "render_sample_thirteen_textures.frag.metal"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8
@__air_sampler_state.1 = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8
@__air_sampler_state.2 = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8
@__air_sampler_state.3 = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8
@__air_sampler_state.4 = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8
@__air_sampler_state.5 = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8
@__air_sampler_state.6 = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8
@__air_sampler_state.7 = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8
@__air_sampler_state.8 = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8
@__air_sampler_state.9 = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8
@__air_sampler_state.10 = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8
@__air_sampler_state.11 = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8
@__air_sampler_state.12 = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8

define <4 x float> @render_sample_thirteen_textures(ptr addrspace(1) readonly captures(none) %tex0, ptr addrspace(1) readonly captures(none) %tex1, ptr addrspace(1) readonly captures(none) %tex2, ptr addrspace(1) readonly captures(none) %tex3, ptr addrspace(1) readonly captures(none) %tex4, ptr addrspace(1) readonly captures(none) %tex5, ptr addrspace(1) readonly captures(none) %tex6, ptr addrspace(1) readonly captures(none) %tex7, ptr addrspace(1) readonly captures(none) %tex8, ptr addrspace(1) readonly captures(none) %tex9, ptr addrspace(1) readonly captures(none) %tex10, ptr addrspace(1) readonly captures(none) %tex11, ptr addrspace(1) readonly captures(none) %tex12) local_unnamed_addr {
entry:
  %pair0 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex0, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 3.750000e-01, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %value0 = extractvalue { <4 x float>, i8 } %pair0, 0
  %red0 = extractelement <4 x float> %value0, i64 0
  %pair1 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex1, ptr addrspace(2) readonly captures(none) @__air_sampler_state.1, <2 x float> <float 3.750000e-01, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %value1 = extractvalue { <4 x float>, i8 } %pair1, 0
  %red1 = extractelement <4 x float> %value1, i64 0
  %pair2 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex2, ptr addrspace(2) readonly captures(none) @__air_sampler_state.2, <2 x float> <float 3.750000e-01, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %value2 = extractvalue { <4 x float>, i8 } %pair2, 0
  %red2 = extractelement <4 x float> %value2, i64 0
  %pair3 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex3, ptr addrspace(2) readonly captures(none) @__air_sampler_state.3, <2 x float> <float 3.750000e-01, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %value3 = extractvalue { <4 x float>, i8 } %pair3, 0
  %red3 = extractelement <4 x float> %value3, i64 0
  %pair4 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex4, ptr addrspace(2) readonly captures(none) @__air_sampler_state.4, <2 x float> <float 3.750000e-01, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %value4 = extractvalue { <4 x float>, i8 } %pair4, 0
  %red4 = extractelement <4 x float> %value4, i64 0
  %pair5 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex5, ptr addrspace(2) readonly captures(none) @__air_sampler_state.5, <2 x float> <float 3.750000e-01, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %value5 = extractvalue { <4 x float>, i8 } %pair5, 0
  %red5 = extractelement <4 x float> %value5, i64 0
  %pair6 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex6, ptr addrspace(2) readonly captures(none) @__air_sampler_state.6, <2 x float> <float 3.750000e-01, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %value6 = extractvalue { <4 x float>, i8 } %pair6, 0
  %red6 = extractelement <4 x float> %value6, i64 0
  %pair7 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex7, ptr addrspace(2) readonly captures(none) @__air_sampler_state.7, <2 x float> <float 3.750000e-01, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %value7 = extractvalue { <4 x float>, i8 } %pair7, 0
  %red7 = extractelement <4 x float> %value7, i64 0
  %pair8 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex8, ptr addrspace(2) readonly captures(none) @__air_sampler_state.8, <2 x float> <float 3.750000e-01, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %value8 = extractvalue { <4 x float>, i8 } %pair8, 0
  %red8 = extractelement <4 x float> %value8, i64 0
  %pair9 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex9, ptr addrspace(2) readonly captures(none) @__air_sampler_state.9, <2 x float> <float 3.750000e-01, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %value9 = extractvalue { <4 x float>, i8 } %pair9, 0
  %red9 = extractelement <4 x float> %value9, i64 0
  %pair10 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex10, ptr addrspace(2) readonly captures(none) @__air_sampler_state.10, <2 x float> <float 3.750000e-01, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %value10 = extractvalue { <4 x float>, i8 } %pair10, 0
  %red10 = extractelement <4 x float> %value10, i64 0
  %pair11 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex11, ptr addrspace(2) readonly captures(none) @__air_sampler_state.11, <2 x float> <float 3.750000e-01, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %value11 = extractvalue { <4 x float>, i8 } %pair11, 0
  %red11 = extractelement <4 x float> %value11, i64 0
  %pair12 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex12, ptr addrspace(2) readonly captures(none) @__air_sampler_state.12, <2 x float> <float 3.750000e-01, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %value12 = extractvalue { <4 x float>, i8 } %pair12, 0
  %red12 = extractelement <4 x float> %value12, i64 0
  %sum01 = fadd fast float %red0, %red1
  %sum012 = fadd fast float %sum01, %red2
  %sum0123 = fadd fast float %sum012, %red3
  %sum01234 = fadd fast float %sum0123, %red4
  %sum012345 = fadd fast float %sum01234, %red5
  %sum0123456 = fadd fast float %sum012345, %red6
  %sum01234567 = fadd fast float %sum0123456, %red7
  %sum012345678 = fadd fast float %sum01234567, %red8
  %sum0123456789 = fadd fast float %sum012345678, %red9
  %sum0123456789a = fadd fast float %sum0123456789, %red10
  %sum0123456789ab = fadd fast float %sum0123456789a, %red11
  %sum0123456789abc = fadd fast float %sum0123456789ab, %red12
  %red_out = insertelement <4 x float> undef, float %sum0123456789abc, i32 0
  %green_out = insertelement <4 x float> %red_out, float %red0, i32 1
  %blue_out = insertelement <4 x float> %green_out, float %red12, i32 2
  %alpha_out = insertelement <4 x float> %blue_out, float 1.000000e+00, i32 3
  ret <4 x float> %alpha_out
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!air.sampler_states = !{!20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32}
!0 = !{ptr @render_sample_thirteen_textures, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex0"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex1"}
!6 = !{i32 2, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex2"}
!7 = !{i32 3, !"air.texture", !"air.location_index", i32 3, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex3"}
!8 = !{i32 4, !"air.texture", !"air.location_index", i32 4, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex4"}
!9 = !{i32 5, !"air.texture", !"air.location_index", i32 5, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex5"}
!10 = !{i32 6, !"air.texture", !"air.location_index", i32 6, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex6"}
!11 = !{i32 7, !"air.texture", !"air.location_index", i32 7, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex7"}
!12 = !{i32 8, !"air.texture", !"air.location_index", i32 8, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex8"}
!13 = !{i32 9, !"air.texture", !"air.location_index", i32 9, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex9"}
!14 = !{i32 10, !"air.texture", !"air.location_index", i32 10, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex10"}
!15 = !{i32 11, !"air.texture", !"air.location_index", i32 11, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex11"}
!16 = !{i32 12, !"air.texture", !"air.location_index", i32 12, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex12"}
!20 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!21 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state.1}
!22 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state.2}
!23 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state.3}
!24 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state.4}
!25 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state.5}
!26 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state.6}
!27 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state.7}
!28 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state.8}
!29 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state.9}
!30 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state.10}
!31 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state.11}
!32 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state.12}
