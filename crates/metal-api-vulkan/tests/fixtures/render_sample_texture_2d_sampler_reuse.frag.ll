; Owned synthetic fixture for the canonical render rail's sampled-sampler-reuse
; arm (2026-09-20; census v48's fifth door, the reims side's R50): one fragment
; stage that carries **two** AIR `constexpr samplers` while **four** sampled
; textures read through them — `[[texture(0)]]`, `[[texture(1)]]` and
; `[[texture(3)]]` through the first state and `[[texture(2)]]` through the
; second.
;
; The shape is the census's own. v48's residual bucket is 26 records × 2 of one
; LPF-family fragment stage (`pipe=58`, `fw=109140`, `fdecl=25`) whose thirteen
; sampled textures read through four sampler descriptors: two runtime
; `[[sampler(n)]]` arguments and the module's own two AIR states, each state
; reused by the textures past the first. The rail's registration paired one AIR
; static sampler with one sampled texture *by position* and refused the counts
; (`render_stage_reflection_mismatch`, "this rail pairs one AIR static sampler
; with one sampled texture"), which is a rule a `constexpr sampler` — a value
; the frontend reuses rather than a slot it spends once — falsifies.
;
; The two states are the corpus's own words, and they are the pair the census
; read (`smpl=4[s832:gNNnee,s833:gNNnee,s834:cLLnee,s835:cNNnrr]`):
; `34901797601018002` is Nearest + Repeat and `34901797601020489` the Linear +
; ClampToEdge state of the sibling `render_sample_texture_2d_linear_clamp.frag.ll`.
; Both are global names the frontend's own numbering gives (`@__air_sampler_state`
; and `@__air_sampler_state.1`), which is the order the translator reads the
; module's AIR sampler states in.
;
; Every texture is sampled at `u = 1.375`, one half column past the surface's
; right edge, which is exactly where those two states part. Against the test's
; 4x4 texture (column `i` holds `64 * i` in red) the fragment lands
; `40 40 c0 40`: the repeated state reads texel `(1, 3)` (`0x40`), the clamped
; linear state texel `(3, 3)` (`0xc0`), so red, green and alpha are the reused
; state's reading, blue the other state's, and a rail that paired the states
; with the textures by position would move green and blue.
;
; Re-assemble with `llvm-as` (22.1.8), never by editing the module by hand:
;
;   llvm-as -o /tmp/render_sample_texture_2d_sampler_reuse.bc \
;           crates/metal-api-vulkan/tests/fixtures/render_sample_texture_2d_sampler_reuse.frag.ll
;
; The vertex half of the pair is the reviewed `render_offscreen_2x2.vert.ll`
; (entry `render_fullscreen_triangle`); the fragment takes no varying, which is
; why the two compose.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.5.0"

source_filename = "render_sample_texture_2d_sampler_reuse.frag.air"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601018002, i64 0], align 8
@__air_sampler_state.1 = internal addrspace(2) constant [2 x i64] [i64 34901797601020489, i64 0], align 8

define <4 x float> @render_sample_texture_2d_sampler_reuse(ptr addrspace(1) readonly captures(none) %repeat_texture, ptr addrspace(1) readonly captures(none) %reuse_texture, ptr addrspace(1) readonly captures(none) %linear_texture, ptr addrspace(1) readonly captures(none) %reuse_texture_b) local_unnamed_addr {
entry:
  %repeat_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %repeat_texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 1.375000e+00, float 8.750000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %repeat_value = extractvalue { <4 x float>, i8 } %repeat_pair, 0
  %repeat = extractelement <4 x float> %repeat_value, i64 0
  %reuse_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %reuse_texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 1.375000e+00, float 8.750000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %reuse_value = extractvalue { <4 x float>, i8 } %reuse_pair, 0
  %reuse = extractelement <4 x float> %reuse_value, i64 0
  %linear_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %linear_texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state.1, <2 x float> <float 1.375000e+00, float 8.750000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %linear_value = extractvalue { <4 x float>, i8 } %linear_pair, 0
  %linear = extractelement <4 x float> %linear_value, i64 0
  %reuse_b_pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %reuse_texture_b, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 1.375000e+00, float 8.750000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %reuse_b_value = extractvalue { <4 x float>, i8 } %reuse_b_pair, 0
  %reuse_b = extractelement <4 x float> %reuse_b_value, i64 0
  %with_red = insertelement <4 x float> undef, float %repeat, i32 0
  %with_green = insertelement <4 x float> %with_red, float %reuse, i32 1
  %with_blue = insertelement <4 x float> %with_green, float %linear, i32 2
  %with_alpha = insertelement <4 x float> %with_blue, float %reuse_b, i32 3
  ret <4 x float> %with_alpha
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr

!air.fragment = !{!0}
!air.sampler_states = !{!6, !8}
!0 = !{ptr @render_sample_texture_2d_sampler_reuse, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4, !5, !9, !10}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"repeat_texture"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"reuse_texture"}
!9 = !{i32 2, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"linear_texture"}
!10 = !{i32 3, !"air.texture", !"air.location_index", i32 3, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"reuse_texture_b"}
!6 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!8 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state.1}
