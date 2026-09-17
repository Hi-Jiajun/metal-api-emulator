; Owned synthetic AIR fixture (E-TX1): one fragment stage that samples four
; texels of its own 4x4 texture and returns one *whole channel* of each, so the
; attachment's four bytes are the texture's four channels read at four texel
; centres.
;
;   red   = texel (0, 0).x     sample at (0.125, 0.125)
;   green = texel (1, 0).y     sample at (0.375, 0.125)
;   blue  = texel (0, 1).z     sample at (0.125, 0.375)
;   alpha = texel (1, 1).w     sample at (0.375, 0.375)
;
; The coordinates are texel centres of a 4-texel-wide surface — (i + 0.5) / 4 —
; so the nearest clamp-to-edge sample is the texel itself rather than a blend,
; and a flipped or transposed coordinate reads another texel's channel (the
; fixture's texture gives every texel its own four values).
;
; The fixture exists for the sampled texture's *byte order*
; (research/docs/23 §107): against a `bgra8_unorm` texture the four bytes above
; are the texel's colours (R, G, B, A), and against the `rgba8_unorm` sibling
; they are the same colours from the same channel indices of the other memory
; layout. A rail that uploaded the bytes under the wrong name would land the
; red and blue halves swapped, which is exactly what the e2e reading pins.
;
; The state is the module's own, exactly as the v100 siblings state it: the
; contract's declaration has to repeat the AIR constexpr sampler state
; (nearest + clamp-to-edge), and a declaration naming another one is refused by
; the rail by name (research/docs/23 §3.3, v100).
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

source_filename = "render_sample_texture_2d_bgra_channels.frag.metal"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8

define <4 x float> @render_sample_texture_2d_bgra_channels(ptr addrspace(1) readonly captures(none) %tex) local_unnamed_addr {
entry:
  %red_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 1.250000e-01, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %red_value = extractvalue { <4 x float>, i8 } %red_pair, 0
  %red = extractelement <4 x float> %red_value, i64 0
  %green_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 3.750000e-01, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %green_value = extractvalue { <4 x float>, i8 } %green_pair, 0
  %green = extractelement <4 x float> %green_value, i64 1
  %blue_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 1.250000e-01, float 3.750000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %blue_value = extractvalue { <4 x float>, i8 } %blue_pair, 0
  %blue = extractelement <4 x float> %blue_value, i64 2
  %alpha_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 3.750000e-01, float 3.750000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %alpha_value = extractvalue { <4 x float>, i8 } %alpha_pair, 0
  %alpha = extractelement <4 x float> %alpha_value, i64 3
  %first = insertelement <4 x float> undef, float %red, i32 0
  %second = insertelement <4 x float> %first, float %green, i32 1
  %third = insertelement <4 x float> %second, float %blue, i32 2
  %fourth = insertelement <4 x float> %third, float %alpha, i32 3
  ret <4 x float> %fourth
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @render_sample_texture_2d_bgra_channels, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
