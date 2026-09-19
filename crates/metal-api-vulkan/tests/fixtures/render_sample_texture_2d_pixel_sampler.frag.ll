; Owned synthetic AIR fixture (E-RS5): one fragment stage that samples a 4x4
; `rgba8_unorm` texture through an AIR-embedded `constexpr sampler` whose only
; out-of-family axis is `coord::pixel`.
;
; The module is the corpus shape census v40 read as
; `render_provider_out_of_class_texture_static_sampler_unpaired`: the Metal
; source wrote a `constexpr sampler` with `coord::pixel`, and the pinned
; translator emulates that sampler with *shader-side image fetches*
; (`43c46ac/src/passes/air_calls/images/sample.rs`: a pixel-coordinate state
; takes the fetch path), so the finished body reads the image with
; `OpImageFetch` and carries no `OpSampledImage` at all — the state the AIR
; metadata still declares is a lowered remnant, not a binding anything reads
; through.
;
; The AIR sampler word is the corpus word `34901797601017929` (nearest /
; nearest, mip `none`, both addressing axes `clampToEdge`, anisotropy 1, LOD
; minimum 0, transparent-black border, weighted average) with bit 15 set,
; which is the `coordinates` field: `34901797601050697`. It differs from the
; reviewed family on that one axis and nowhere else, so the fixture separates
; "the AIR state nothing reads" from "a state that changed what the module
; executes".
;
; The two sites sample the *texel centres* (1.5, 0.5) and (0.5, 1.5); the
; pixel-coordinate emulation floors a coordinate to its texel. Against a
; texture whose texel `(i, j)` holds `(64 * i, 64 * j, 0, 255)` the two reads
; land
;
;   red   = texel (1, 0).x = 64  -> 40
;   green = texel (0, 1).y = 64  -> 40
;
; so the frame is `40 40 00 ff` for every covered fragment — byte-for-byte the
; frame the sampler-free sibling fixture (`render_fetch_texture_2d.frag.ll`)
; lands from the same texture bytes, which is what makes "the AIR state did not
; participate" an observation rather than a claim.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

source_filename = "render_sample_texture_2d_pixel_sampler.frag.metal"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601050697, i64 0], align 8

define <4 x float> @render_sample_texture_2d_pixel_sampler(ptr addrspace(1) readonly captures(none) %tex) local_unnamed_addr {
entry:
  %left_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 1.500000e+00, float 5.000000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %left_value = extractvalue { <4 x float>, i8 } %left_pair, 0
  %left = extractelement <4 x float> %left_value, i64 0
  %right_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 5.000000e-01, float 1.500000e+00>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %right_value = extractvalue { <4 x float>, i8 } %right_pair, 0
  %right = extractelement <4 x float> %right_value, i64 1
  %red = insertelement <4 x float> undef, float %left, i32 0
  %green = insertelement <4 x float> %red, float %right, i32 1
  %blue = insertelement <4 x float> %green, float 0.000000000000000e+00, i32 2
  %alpha = insertelement <4 x float> %blue, float 1.000000e+00, i32 3
  ret <4 x float> %alpha
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @render_sample_texture_2d_pixel_sampler, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
