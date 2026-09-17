; Owned synthetic AIR fixture (E-RS2): one fragment stage whose two sampled
; textures each read through their *own runtime* `[[sampler(n)]]` argument.
;
; This is the shape the census found blocking (`texture_interface` /
; `texture_sampler`): the module carries no `@__air_sampler_state` and no
; `air.sampler_states` root at all, because Metal binds the sampler object when
; the draw is encoded — the state is a *request* fact, and the pass states it.
;
; The body samples two 4x4 `rgba8_unorm` textures at normalized coordinates and
; stores three readings as one colour, so both textures and both samplers are
; observable in the attachment's bytes:
;
;   red   = left.sample(sampler 0, (1.375, 0.125)).x
;   green = right.sample(sampler 1, (1.375, 0.125)).x
;   blue  = right.sample(sampler 1, (0.3125, 0.125)).x
;   alpha = 1.0
;
; Against a texture whose column `i` holds `64 * i` in red, the states land:
;
;   sampler 0 nearest+clamp, sampler 1 nearest+clamp: c0 c0 40 ff
;   sampler 0 nearest+repeat, sampler 1 nearest+repeat: 40 40 40 ff
;   sampler 0 nearest+clamp, sampler 1 linear+clamp:  c0 c0 30 ff
;
; `u = 1.375` is the coordinate that separates clamp-to-edge (the edge texel,
; 192) from repeat (texel 1, 64); `u = 0.3125` is the one that separates
; nearest (texel 1, 64) from linear (a quartile of texel 0 mixed in, 48). Every
; reading is a multiple of sixteen, so the 8-bit unorm quantization of a linear
; blend is exact on any driver that filters linearly at all.
;
; The sampler state is the *pass's*: the registration pairs each texture with
; the runtime sampler index it reads through, and the rail creates the
; descriptor's `VkSampler` from the state the pass states
; (research/docs/23 §3.3, v102).
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

source_filename = "render_sample_texture_2d_runtime.frag.metal"

define <4 x float> @render_sample_texture_2d_runtime(ptr addrspace(1) readonly captures(none) %left, ptr addrspace(1) readonly captures(none) %right, ptr addrspace(2) readonly captures(none) %first_sampler, ptr addrspace(2) readonly captures(none) %second_sampler) local_unnamed_addr {
entry:
  %edge_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %left, ptr addrspace(2) readonly captures(none) %first_sampler, <2 x float> <float 1.375000e+00, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %edge_value = extractvalue { <4 x float>, i8 } %edge_pair, 0
  %edge = extractelement <4 x float> %edge_value, i64 0
  %wrapped_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %right, ptr addrspace(2) readonly captures(none) %second_sampler, <2 x float> <float 1.375000e+00, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %wrapped_value = extractvalue { <4 x float>, i8 } %wrapped_pair, 0
  %wrapped = extractelement <4 x float> %wrapped_value, i64 0
  %filtered_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %right, ptr addrspace(2) readonly captures(none) %second_sampler, <2 x float> <float 3.125000e-01, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %filtered_value = extractvalue { <4 x float>, i8 } %filtered_pair, 0
  %filtered = extractelement <4 x float> %filtered_value, i64 0
  %green = insertelement <4 x float> undef, float %edge, i32 0
  %blue = insertelement <4 x float> %green, float %wrapped, i32 1
  %filled = insertelement <4 x float> %blue, float %filtered, i32 2
  %alpha = insertelement <4 x float> %filled, float 1.000000e+00, i32 3
  ret <4 x float> %alpha
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!0 = !{ptr @render_sample_texture_2d_runtime, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4, !5, !6, !7}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"left"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"right"}
!6 = !{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"first_sampler"}
!7 = !{i32 3, !"air.sampler", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"second_sampler"}
