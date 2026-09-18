; Owned synthetic AIR fixture (E-TX2): one fragment stage whose one sampled
; texture reads through one *runtime* `[[sampler(n)]]` argument at three
; coordinates that separate the address modes the canonical family names
; (research/docs/23 §109).
;
; The module carries no `@__air_sampler_state`: Metal binds the sampler object
; when the draw is encoded, so the state is a *request* fact and the pass
; states it. The body samples one 4x4 `rgba8_unorm` texture at
;
;   red   = tex.sample(sampler 0, (-0.25, 0.125)).x
;   green = tex.sample(sampler 0, ( 1.25, 0.125)).x
;   blue  = tex.sample(sampler 0, ( 2.25, 0.125)).x
;   alpha = 1.0
;
; Against a texture whose column `i` holds `64 * i` in red — the four texels
; are 00, 40, 80, c0 — nearest filtering and the five address modes land:
;
;   clamp-to-edge        00 c0 c0 ff   (-0.25 clamps to 0; 1.25 and 2.25 to 1)
;   mirror-clamp-to-edge 40 c0 c0 ff   (|-0.25| = 0.25; the other two clamp)
;   repeat               c0 40 40 ff   (wraps to 0.75, then 0.25 and 0.25)
;   mirror-repeat        40 c0 40 ff   (0.25, 0.75, 0.25)
;   clamp-to-zero        00 00 00 ff   (all three read the zero border)
;
; Every reading is a multiple of sixteen, so the 8-bit unorm quantization of a
; linear blend is exact on any driver that filters linearly at all.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

source_filename = "render_sample_texture_2d_boundary.frag.metal"

define <4 x float> @render_sample_texture_2d_boundary(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) %boundary_sampler) local_unnamed_addr {
entry:
  %edge_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) %boundary_sampler, <2 x float> <float -2.500000e-01, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %edge_value = extractvalue { <4 x float>, i8 } %edge_pair, 0
  %edge = extractelement <4 x float> %edge_value, i64 0
  %edge_pair_right = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) %boundary_sampler, <2 x float> <float 1.250000e+00, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %edge_right_value = extractvalue { <4 x float>, i8 } %edge_pair_right, 0
  %edge_right = extractelement <4 x float> %edge_right_value, i64 0
  %far_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) %boundary_sampler, <2 x float> <float 2.250000e+00, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %far_value = extractvalue { <4 x float>, i8 } %far_pair, 0
  %far = extractelement <4 x float> %far_value, i64 0
  %green = insertelement <4 x float> undef, float %edge, i32 0
  %blue = insertelement <4 x float> %green, float %edge_right, i32 1
  %filled = insertelement <4 x float> %blue, float %far, i32 2
  %alpha = insertelement <4 x float> %filled, float 1.000000e+00, i32 3
  ret <4 x float> %alpha
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!0 = !{ptr @render_sample_texture_2d_boundary, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4, !5}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!5 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"boundary_sampler"}
