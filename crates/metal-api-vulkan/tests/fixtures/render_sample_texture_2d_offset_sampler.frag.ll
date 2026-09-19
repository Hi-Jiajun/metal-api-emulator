; Owned synthetic AIR fixture (2026-09-19, census v43's `texture_state` axis):
; the runtime-sampler fixture one axis over — every sample carries a **non-zero
; constant offset**.
;
; The offset is what an unnormalized `VkSampler` may not be used with
; (`VUID-vkCmdDraw-None-08611`), so this module has no explicit-LOD sibling:
; the rail answers a pass that binds the texel space to it by name
; (`render_pixel_sampler_variant_unavailable`) instead of executing the sample
; form the API forbids. The module is otherwise the runtime fixture, so the
; same registration and pass shape exercises both the normalized arm (which
; executes, with the offset the translator emits) and the texel-space refusal.
;
; Body, against `rgba8_unorm` textures whose column `i` holds `64 * i` in red:
;
;   red   = left.sample(sampler 0, (1.375, 0.125), offset (1, 0)).x
;   green = right.sample(sampler 1, (1.375, 0.125), offset (1, 0)).x
;   blue  = right.sample(sampler 1, (0.3125, 0.125), offset (1, 0)).x
;   alpha = 1.0
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

source_filename = "render_sample_texture_2d_offset_sampler.frag.metal"

define <4 x float> @render_sample_texture_2d_offset_sampler(ptr addrspace(1) readonly captures(none) %left, ptr addrspace(1) readonly captures(none) %right, ptr addrspace(2) readonly captures(none) %first_sampler, ptr addrspace(2) readonly captures(none) %second_sampler) local_unnamed_addr {
entry:
  %edge_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %left, ptr addrspace(2) readonly captures(none) %first_sampler, <2 x float> <float 1.375000e+00, float 1.250000e-01>, i1 true, <2 x i32> <i32 1, i32 0>, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %edge_value = extractvalue { <4 x float>, i8 } %edge_pair, 0
  %edge = extractelement <4 x float> %edge_value, i64 0
  %wrapped_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %right, ptr addrspace(2) readonly captures(none) %second_sampler, <2 x float> <float 1.375000e+00, float 1.250000e-01>, i1 true, <2 x i32> <i32 1, i32 0>, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %wrapped_value = extractvalue { <4 x float>, i8 } %wrapped_pair, 0
  %wrapped = extractelement <4 x float> %wrapped_value, i64 0
  %filtered_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %right, ptr addrspace(2) readonly captures(none) %second_sampler, <2 x float> <float 3.125000e-01, float 1.250000e-01>, i1 true, <2 x i32> <i32 1, i32 0>, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
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
!0 = !{ptr @render_sample_texture_2d_offset_sampler, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4, !5, !6, !7}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"left"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"right"}
!6 = !{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"first_sampler"}
!7 = !{i32 3, !"air.sampler", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"second_sampler"}
