; Owned synthetic AIR fixture (E-TX7): one fragment stage that reads *both* a
; `[[texture(0)]]` argument through an AIR constexpr sampler and a
; `[[buffer(0)]]` stage argument, which is the shape the translator's own
; descriptor layout puts in the one set 0 — the buffer band `0..32` and the
; sampled-texture band `32..160` side by side, so the two faces share one
; layout, one pool and one set instead of one of them being dropped.
;
; The body is `render_sample_texture_2d_nearest_clamp.frag.ll`'s first sample
; plus `render_stage_buffer_tint.frag.ll`'s one `float4` load, and the colour
; it returns keeps the two reads in separate channels so each one is visible on
; its own: red is the texel the module's own nearest + clamp-to-edge sampler
; returns at (1.375, 0.125) — the coordinate outside the unit square, so the
; clamped reading is the texture's own edge texel (192 against a texture whose
; column `i` holds `64 * i` in red) — green and blue are the tint's second and
; third components. Swapping either payload's bytes therefore moves the frame
; in the channels that payload owns and in no others.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

source_filename = "render_sample_texture_2d_stage_buffer_tint.frag.metal"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8

define <4 x float> @render_sample_texture_2d_stage_buffer_tint(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(1) readonly captures(none) %tint) local_unnamed_addr {
entry:
  %addressed_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 1.375000e+00, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %addressed_value = extractvalue { <4 x float>, i8 } %addressed_pair, 0
  %sampled = extractelement <4 x float> %addressed_value, i64 0
  %tint_value = load <4 x float>, ptr addrspace(1) %tint, align 16
  %tint_green = extractelement <4 x float> %tint_value, i64 1
  %tint_blue = extractelement <4 x float> %tint_value, i64 2
  %red = insertelement <4 x float> undef, float %sampled, i32 0
  %green = insertelement <4 x float> %red, float %tint_green, i32 1
  %blue = insertelement <4 x float> %green, float %tint_blue, i32 2
  %alpha = insertelement <4 x float> %blue, float 1.000000e+00, i32 3
  ret <4 x float> %alpha
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @render_sample_texture_2d_stage_buffer_tint, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4, !6}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!6 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"tint"}
