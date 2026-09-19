; Owned synthetic AIR fixture (E-RS5): the *bound* sibling of
; `render_sample_texture_2d_pixel_sampler.frag.ll` — one fragment stage that
; samples a 4x4 `rgba8_unorm` texture through its own AIR-embedded
; `constexpr sampler`, whose state is outside the reviewed family on the
; min/mag filter axis (Nearest minification, Linear magnification).
;
; The sampler state is the corpus word `34901797601017929` (nearest / nearest,
; mip `none`, both addressing axes `clampToEdge`, anisotropy 1, LOD minimum 0,
; transparent-black border, weighted average) with the magnification filter
; raised to `linear`, which is the field at bits 9-10: `34901797601018441`.
; The coordinates stay normalized, so the pinned translator lowers this sample
; to a genuine `OpSampledImage` — the module really reads through this sampler,
; and a rail that bypassed its state would execute a filter the module named
; and did not get.
;
; The registration is therefore the control beside the pixel-coordinate
; fixture: the same two sample sites, the same declaration shape, one axis
; moved — the bound state is weighed and refused by name
; (`render_stage_unsupported_interface`), while the unread state is not.
; This fixture lands no frame: the refusal happens before any pass can run.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

source_filename = "render_sample_texture_2d_mixed_filters.frag.metal"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601018441, i64 0], align 8

define <4 x float> @render_sample_texture_2d_mixed_filters(ptr addrspace(1) readonly captures(none) %tex) local_unnamed_addr {
entry:
  %left_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 1.375000e+00, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %left_value = extractvalue { <4 x float>, i8 } %left_pair, 0
  %left = extractelement <4 x float> %left_value, i64 0
  %right_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 3.125000e-01, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %right_value = extractvalue { <4 x float>, i8 } %right_pair, 0
  %right = extractelement <4 x float> %right_value, i64 0
  %red = insertelement <4 x float> undef, float %left, i32 0
  %green = insertelement <4 x float> %red, float %right, i32 1
  %blue = insertelement <4 x float> %green, float 0.000000000000000e+00, i32 2
  %alpha = insertelement <4 x float> %blue, float 1.000000e+00, i32 3
  ret <4 x float> %alpha
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @render_sample_texture_2d_mixed_filters, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
