; Owned synthetic AIR fixture (E-RS4): one fragment stage that reads a 4x4
; `rgba8_unorm` texture with `texture.read()` — Metal's `access::read`
; qualifier — and stores one column's red and one row's green.
;
; This is the shape the census v11 read as "sample sites name no runtime
; sampler" (`research/docs/23` §3.3, v105): the module carries no
; `@__air_sampler_state`, no `air.sampler_states` root and no
; `[[sampler(n)]]` argument, because an `access::read` texture is fetched by
; integer coordinate at an explicit level of detail rather than sampled. The
; translator lowers this body to exactly one `OpImageFetch` per read and no
; `OpSampledImage` at all, so the descriptor the rail binds is the image alone.
;
; Against a texture whose texel `(i, j)` holds `(64 * i, 64 * j, 0, 255)`, the
; two reads land
;
;   red   = texel (1, 0).x = 64  -> 40
;   green = texel (0, 1).y = 64  -> 40
;
; so the frame is `40 40 00 ff` for every covered fragment. Every reading is a
; multiple of sixteen, so the 8-bit unorm quantisation is exact on any driver,
; and the two halves move with different coordinates, so both axes of the integer
; coordinate are observed.
;
; The fetch is the *module's* own statement: the pass declares the binding
; `Fetched`, and a declaration that says `Sampled` there is refused by the rail
; by name (`render_texture_access_unsupported`).
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

source_filename = "render_fetch_texture_2d.frag.metal"

define <4 x float> @render_fetch_texture_2d(ptr addrspace(1) readonly captures(none) %tex) local_unnamed_addr {
entry:
  %left_pair = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex, <2 x i32> <i32 1, i32 0>, i32 0, i32 0)
  %left_value = extractvalue { <4 x float>, i8 } %left_pair, 0
  %left = extractelement <4 x float> %left_value, i64 0
  %right_pair = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %tex, <2 x i32> <i32 0, i32 1>, i32 0, i32 0)
  %right_value = extractvalue { <4 x float>, i8 } %right_pair, 0
  %right = extractelement <4 x float> %right_value, i64 1
  %red = insertelement <4 x float> undef, float %left, i32 0
  %green = insertelement <4 x float> %red, float %right, i32 1
  %blue = insertelement <4 x float> %green, float 0.000000000000000e+00, i32 2
  %alpha = insertelement <4 x float> %blue, float 1.000000e+00, i32 3
  ret <4 x float> %alpha
}

declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), <2 x i32>, i32, i32) local_unnamed_addr

!air.fragment = !{!0}
!0 = !{ptr @render_fetch_texture_2d, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"tex"}
