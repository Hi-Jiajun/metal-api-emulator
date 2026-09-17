; Owned synthetic fixture (C1b): byte-for-byte the same kernel body as
; `sample_texture_2d_nearest_clamp.ll` except for the AIR-embedded constexpr
; sampler state, which repeats instead of clamping. Not derived from a
; third-party metallib.
;
; One invocation samples a 4x4 `R32Float` texture twice and stores both
; component-zero values into one 8-byte `float2` output buffer:
;
;   sample A: (1.375, 0.125)
;       - clamp-to-edge: u=1.375 clamps to the texture edge (texel 3)
;       - repeat:        u=1.375 wraps to 0.375, the centre of texel 1
;   sample B: (0.3125, 0.125)
;       - nearest: the sample point falls in texel 1
;       - linear:  quartile blend between texel 0 and texel 1
;
; Against a row of `[0, 4, 8, 12]` along x and `y = 0.125` (row 0's centre) the
; two readings therefore separate all four states the C1b increment names:
; nearest+clamp = [12, 4], nearest+repeat = [4, 4], linear+clamp = [12, 3].
source_filename = "kernel_sample_texture_2d_nearest_repeat.metal"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601018002, i64 0], align 8

define void @sample_texture_2d(ptr addrspace(1) readonly captures(none) %texture, ptr addrspace(1) noundef writeonly captures(none) %output) local_unnamed_addr {
entry:
  %first = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 1.375000e+00, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %first_value = extractvalue { <4 x float>, i8 } %first, 0
  %addressed = extractelement <4 x float> %first_value, i64 0
  %second = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %texture, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> <float 3.125000e-01, float 1.250000e-01>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %second_value = extractvalue { <4 x float>, i8 } %second, 0
  %filtered = extractelement <4 x float> %second_value, i64 0
  %pair0 = insertelement <2 x float> undef, float %addressed, i64 0
  %pair = insertelement <2 x float> %pair0, float %filtered, i64 1
  store <2 x float> %pair, ptr addrspace(1) %output, align 8
  ret void
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr

!air.kernel = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @sample_texture_2d, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"out"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
