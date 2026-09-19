; Owned synthetic AIR fixture (2026-09-20, the `D3` sampled texture arm): one
; fragment stage whose `[[texture(0)]]` is a **three-dimensional volume** —
; `texture3d<float, sample>` — sampled at four texel centres of its own
; `4 x 4 x 2` grid.
;
;   red   = slice 0, (0, 0)   sample at (0.125, 0.125, 0.25)
;   green = slice 1, (2, 0)   sample at (0.625, 0.125, 0.75)
;   blue  = slice 0, (0, 2)   sample at (0.125, 0.625, 0.25)
;   alpha = slice 1, (2, 2)   sample at (0.625, 0.625, 0.75)
;
; The coordinates are texel centres of a `4 x 4 x 2` volume — `(i + 0.5) / 4`
; in x and y and `(k + 0.5) / 2` in z — so the nearest *and* the linear sample
; of that point is the texel itself rather than a blend: both weights of the
; 3D filter are exactly one and zero, which is what makes the frame a function
; of the upload rather than of a driver's filtering precision.
;
; The four *z* coordinates are the arm's own statement. A rail that read the
; volume as a two-dimensional grid of *layers* — `depth` as an array axis, so
; the third coordinate selected a slice of a `4 x 4 x 2` array rather than the
; volume's own third axis — lands the same texels only if the two happen to
; agree, and a rail that uploaded only the first slice lands the fill value in
; the two lanes the fixture samples out of slice 1. The four positions differ
; in x and y as well, so a transposed or row-shifted read lands another texel's
; number too.
;
; Every sample takes `.x` because a single-component texture carries exactly
; one channel: Vulkan fills the components an `R32_SFLOAT` view does not have
; (green and blue zero, alpha one), so a fixture reading `.y` would read the
; fill rather than the volume.
;
; The state is the module's own, exactly as the v100 siblings state it: the
; contract's declaration has to repeat the AIR constexpr sampler state (linear
; + clamp-to-edge), and a declaration naming another one is refused by the rail
; by name (research/docs/23 §3.3, v100).
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

source_filename = "render_sample_texture_3d_volume.frag.metal"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601020489, i64 0], align 8

define <4 x float> @render_sample_texture_3d_volume(ptr addrspace(1) readonly captures(none) %tex) local_unnamed_addr {
entry:
  %red_pair = tail call { <4 x float>, i8 } @air.sample_texture_3d.v4f32(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <3 x float> <float 1.250000e-01, float 1.250000e-01, float 2.500000e-01>, i1 true, <3 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %red_value = extractvalue { <4 x float>, i8 } %red_pair, 0
  %red = extractelement <4 x float> %red_value, i64 0
  %green_pair = tail call { <4 x float>, i8 } @air.sample_texture_3d.v4f32(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <3 x float> <float 6.250000e-01, float 1.250000e-01, float 7.500000e-01>, i1 true, <3 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %green_value = extractvalue { <4 x float>, i8 } %green_pair, 0
  %green = extractelement <4 x float> %green_value, i64 0
  %blue_pair = tail call { <4 x float>, i8 } @air.sample_texture_3d.v4f32(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <3 x float> <float 1.250000e-01, float 6.250000e-01, float 2.500000e-01>, i1 true, <3 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %blue_value = extractvalue { <4 x float>, i8 } %blue_pair, 0
  %blue = extractelement <4 x float> %blue_value, i64 0
  %alpha_pair = tail call { <4 x float>, i8 } @air.sample_texture_3d.v4f32(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <3 x float> <float 6.250000e-01, float 6.250000e-01, float 7.500000e-01>, i1 true, <3 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %alpha_value = extractvalue { <4 x float>, i8 } %alpha_pair, 0
  %alpha = extractelement <4 x float> %alpha_value, i64 0
  %first = insertelement <4 x float> undef, float %red, i32 0
  %second = insertelement <4 x float> %first, float %green, i32 1
  %third = insertelement <4 x float> %second, float %blue, i32 2
  %fourth = insertelement <4 x float> %third, float %alpha, i32 3
  ret <4 x float> %fourth
}

declare { <4 x float>, i8 } @air.sample_texture_3d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <3 x float>, i1, <3 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @render_sample_texture_3d_volume, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture3d<float, sample>", !"air.arg_name", !"tex"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
