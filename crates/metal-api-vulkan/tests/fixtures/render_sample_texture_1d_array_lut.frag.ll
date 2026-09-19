; Owned synthetic AIR fixture (2026-09-19, census b10's `texture_shape`
; bucket): one fragment stage whose `[[texture(0)]]` is a **one-dimensional
; array** LUT — `texture1d_array<float, sample>` — sampled at four texel
; centres of its own single row.
;
;   red   = texel 0.x     sample at u = 0.0625
;   green = texel 2.x     sample at u = 0.3125
;   blue  = texel 4.x     sample at u = 0.5625
;   alpha = texel 6.x     sample at u = 0.8125
;
; The coordinates are texel centres of an eight-texel row — (i + 0.5) / 8 — so
; the nearest clamp-to-edge sample is the texel itself rather than a blend, and
; a flipped or transposed coordinate reads another texel's value (the fixture's
; LUT gives every reached texel its own number).
;
; Every sample takes `.x` because a single-component texture carries exactly
; one channel: Vulkan fills the components an `R32_SFLOAT`/`R16_SFLOAT` view
; does not have (green and blue zero, alpha one), so a fixture reading `.y`
; would read the fill rather than the LUT. The four *positions* are what make
; the frame falsifiable instead.
;
; The fixture exists for the sampled texture's *dimensionality*
; (`research/docs/23` §119): the census's 215 records are single-row float LUTs
; — a `16384x1` `R32_SFLOAT` colour-transfer table and a `1024x1` `R16_SFLOAT`
; one — bound as `MTLTextureType1DArray` views and declared as
; `texture1d_array<float, sample>`. A rail that created a 2D image for this
; declaration, or a `TYPE_1D` view where the module's coordinate is a
; `float2` of `(u, layer)`, would sample another texel or fail the descriptor
; pairing outright — which is exactly what the e2e reading pins. The layer
; index is the array's own second coordinate and is `0`: the census's LUTs are
; one-slice arrays, and the contract admits exactly that shape.
;
; The state is the module's own, exactly as the v100 siblings state it: the
; contract's declaration has to repeat the AIR constexpr sampler state
; (nearest + clamp-to-edge), and a declaration naming another one is refused by
; the rail by name (research/docs/23 §3.3, v100).
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

source_filename = "render_sample_texture_1d_array_lut.frag.metal"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8

define <4 x float> @render_sample_texture_1d_array_lut(ptr addrspace(1) readonly captures(none) %tex) local_unnamed_addr {
entry:
  %red_pair = tail call { <4 x float>, i8 } @air.sample_texture_1d_array.v4f32(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) @__air_sampler_state, float 6.250000e-02, i32 0, i1 false, i32 0, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %red_value = extractvalue { <4 x float>, i8 } %red_pair, 0
  %red = extractelement <4 x float> %red_value, i64 0
  %green_pair = tail call { <4 x float>, i8 } @air.sample_texture_1d_array.v4f32(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) @__air_sampler_state, float 3.125000e-01, i32 0, i1 false, i32 0, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %green_value = extractvalue { <4 x float>, i8 } %green_pair, 0
  %green = extractelement <4 x float> %green_value, i64 0
  %blue_pair = tail call { <4 x float>, i8 } @air.sample_texture_1d_array.v4f32(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) @__air_sampler_state, float 5.625000e-01, i32 0, i1 false, i32 0, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %blue_value = extractvalue { <4 x float>, i8 } %blue_pair, 0
  %blue = extractelement <4 x float> %blue_value, i64 0
  %alpha_pair = tail call { <4 x float>, i8 } @air.sample_texture_1d_array.v4f32(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) @__air_sampler_state, float 8.125000e-01, i32 0, i1 false, i32 0, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %alpha_value = extractvalue { <4 x float>, i8 } %alpha_pair, 0
  %alpha = extractelement <4 x float> %alpha_value, i64 0
  %first = insertelement <4 x float> undef, float %red, i32 0
  %second = insertelement <4 x float> %first, float %green, i32 1
  %third = insertelement <4 x float> %second, float %blue, i32 2
  %fourth = insertelement <4 x float> %third, float %alpha, i32 3
  ret <4 x float> %fourth
}

declare { <4 x float>, i8 } @air.sample_texture_1d_array.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), float, i32, i1, i32, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @render_sample_texture_1d_array_lut, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture1d_array<float, sample>", !"air.arg_name", !"tex"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
