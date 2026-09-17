; Owned synthetic AIR fixture (E-RS1): the same fragment body as its two
; siblings (`render_sample_texture_2d_nearest_repeat.frag.ll`,
; `render_sample_texture_2d_linear_clamp.frag.ll`), differing only in the
; AIR-embedded constexpr sampler state (Nearest + ClampToEdge).
;
; One fragment stage samples a 4x4 `rgba8_unorm` texture twice and returns both
; component-zero readings as one colour: red is the sample at (1.375, 0.125),
; which separates clamp-to-edge (the edge texel) from repeat (texel 1), and
; green is the sample at (0.3125, 0.125), which separates nearest (texel 1) from
; linear (a quartile of texel 0 mixed in). Against a texture whose column `i`
; holds `64 * i` in the red channel the three states land:
;
;   nearest + clamp-to-edge: 192, 64  -> c0 40 00 ff
;   nearest + repeat:         64, 64  -> 40 40 00 ff
;   linear + clamp-to-edge:  192, 48  -> c0 30 00 ff
;
; Every reading is a multiple of sixteen, so the 8-bit unorm quantization of the
; blend is exact on any driver that filters linearly at all.
;
; The state is the *module's own*: the contract's declaration has to repeat it,
; and a declaration naming another one is refused by the rail by name
; (research/docs/23 §3.3, v100).
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

source_filename = "render_sample_texture_2d_nearest_clamp.frag.metal"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8

define <4 x float> @render_sample_texture_2d(ptr addrspace(1) readonly captures(none) %tex) local_unnamed_addr {
entry:
  %addressed_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 1.375000e+00, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %addressed_value = extractvalue { <4 x float>, i8 } %addressed_pair, 0
  %addressed = extractelement <4 x float> %addressed_value, i64 0
  %filtered_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 3.125000e-01, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %filtered_value = extractvalue { <4 x float>, i8 } %filtered_pair, 0
  %filtered = extractelement <4 x float> %filtered_value, i64 0
  %red = insertelement <4 x float> undef, float %addressed, i32 0
  %green = insertelement <4 x float> %red, float %filtered, i32 1
  %blue = insertelement <4 x float> %green, float 0.000000000000000e+00, i32 2
  %alpha = insertelement <4 x float> %blue, float 1.000000e+00, i32 3
  ret <4 x float> %alpha
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @render_sample_texture_2d, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
