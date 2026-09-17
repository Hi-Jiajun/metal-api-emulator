; Owned synthetic AIR fixture (E-RS3): the same fragment body as
; `render_sample_texture_2d_nearest_clamp.frag.ll`, with the one and only
; `[[texture(n)]]` argument sitting at Metal index **3** rather than 0.
;
; This is the census v11 shape the render contract could not state before
; `v104`: 49.2% of that boot's first-failure lines were a fragment stage whose
; sampled texture is `[[texture(3)]]` while the draw bound it there — the
; contract's texture list used to be *positional*, so a list that had to skip an
; index had no way to say what it meant, and the draw stayed on the engine.
;
; The body is the reviewed static-sampler one: one `rgba8_unorm` 4x4 texture
; sampled twice at fixed coordinates, both component-zero readings returned as
; one colour (red = the clamped sample at (1.375, 0.125), green = the sample at
; (0.3125, 0.125)) through the module's own AIR constexpr sampler
; (nearest + clamp-to-edge). Against a texture whose column `i` holds `64 * i`
; in red the frame is `c0 40 00 ff`, exactly the frame the index-0 sibling
; lands: the *index* moves, the texels do not.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

source_filename = "render_sample_texture_2d_index3.frag.metal"

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
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 3, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
