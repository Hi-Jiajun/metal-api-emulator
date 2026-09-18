; Owned synthetic AIR fixture (sampler-family falsification probe): one
; fragment stage that carries *both* sampler forms at once — one texture is
; sampled through the module's own AIR `constexpr sampler`
; (`@__air_sampler_state`), the other through a runtime `[[sampler(n)]]`
; argument whose state only the pass states.
;
; The census's `texture_sampler_family` bucket is exactly this shape: a stage
; with one AIR static sampler beside one runtime `[[sampler(n)]]`. The question
; this fixture answers is whether the rail executes it or refuses it by name,
; and it is written so that both halves are separately observable in the
; attachment's bytes:
;
;   red   = static_texture.sample(constexpr sampler, (1.375, 0.125)).x
;   green = runtime_texture.sample(sampler 0,     (1.375, 0.125)).x
;   blue  = runtime_texture.sample(sampler 0,     (0.3125, 0.125)).x
;   alpha = 1.0
;
; Against a texture whose column `i` holds `64 * i` in red, and with the
; module's own state Nearest + ClampToEdge:
;
;   runtime sampler nearest + clamp: c0 c0 40 ff — both 1.375 readings clamp
;     to the edge texel, the 0.3125 reading lands in texel 1;
;   runtime sampler nearest + repeat: c0 40 40 ff — the runtime half wraps to
;     texel 1 while the static half stays on the edge;
;   runtime sampler linear + clamp: c0 c0 30 ff — the runtime half blends a
;     quartile of texel 0 into texel 1.
;
; The red channel therefore isolates the *static* half and the green/blue
; channels isolate the *runtime* half: a rail that silently resolved one form
; through the other would move the red channel away from `c0`, which is the
; reading the static-only sibling fixture lands for the same sample.
;
; Every reading is a multiple of sixteen, so the 8-bit unorm quantization of a
; linear blend is exact on any driver that filters linearly at all.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

source_filename = "render_sample_texture_2d_static_and_runtime.frag.metal"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8

define <4 x float> @render_sample_texture_2d_mixed(ptr addrspace(1) readonly captures(none) %static_texture, ptr addrspace(1) readonly captures(none) %runtime_texture, ptr addrspace(2) readonly captures(none) %runtime_sampler) local_unnamed_addr {
entry:
  %static_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %static_texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 1.375000e+00, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %static_value = extractvalue { <4 x float>, i8 } %static_pair, 0
  %static = extractelement <4 x float> %static_value, i64 0
  %addressed_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %runtime_texture, ptr addrspace(2) readonly captures(none) %runtime_sampler, <2 x float> <float 1.375000e+00, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %addressed_value = extractvalue { <4 x float>, i8 } %addressed_pair, 0
  %addressed = extractelement <4 x float> %addressed_value, i64 0
  %filtered_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %runtime_texture, ptr addrspace(2) readonly captures(none) %runtime_sampler, <2 x float> <float 3.125000e-01, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %filtered_value = extractvalue { <4 x float>, i8 } %filtered_pair, 0
  %filtered = extractelement <4 x float> %filtered_value, i64 0
  %green = insertelement <4 x float> undef, float %static, i32 0
  %blue = insertelement <4 x float> %green, float %addressed, i32 1
  %filled = insertelement <4 x float> %blue, float %filtered, i32 2
  %alpha = insertelement <4 x float> %filled, float 1.000000e+00, i32 3
  ret <4 x float> %alpha
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!air.sampler_states = !{!6}
!0 = !{ptr @render_sample_texture_2d_mixed, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4, !5, !7}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"static_texture"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"runtime_texture"}
!7 = !{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"runtime_sampler"}
!6 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
