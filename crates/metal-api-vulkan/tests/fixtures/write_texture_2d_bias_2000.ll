; Owned synthetic fixture (C2): a kernel that writes one `R32Float` storage-image
; texel per invocation through `air.write_texture_2d`. Not derived from a
; third-party metallib.
;
; One invocation writes `float(4 * y + x) + bias` into the texel at
; `(thread_position_in_grid.x, thread_position_in_grid.y)` of a 4x4
; `texture2d<float, access::write>`. The pair of files differs in exactly two
; lines — the `fadd` bias below and the `source_filename` line above it — so a
; 4x4 view has to read back the little-endian floats `[bias, bias + 1, ...,
; bias + 15]`. The pattern is the dispatch's own coordinate arithmetic, so a
; landing that misses a row, a texel, or the kernel's own value cannot agree
; with it, and a readback that ignores which module ran cannot pass both files.
source_filename = "kernel_write_texture_2d_bias_2000.metal"

define void @write_texture_2d(ptr addrspace(1) noundef writeonly captures(none) %texture, <3 x i32> %gid) local_unnamed_addr {
entry:
  %x = extractelement <3 x i32> %gid, i32 0
  %y = extractelement <3 x i32> %gid, i32 1
  %coord0 = insertelement <2 x i32> undef, i32 %x, i64 0
  %coord = insertelement <2 x i32> %coord0, i32 %y, i64 1
  %linear = shl i32 %y, 2
  %cell = add i32 %linear, %x
  %cellf = sitofp i32 %cell to float
  %value = fadd float %cellf, 2.000000e+03
  %texel0 = insertelement <4 x float> undef, float %value, i64 0
  %texel1 = insertelement <4 x float> %texel0, float 0.000000e+00, i64 1
  %texel2 = insertelement <4 x float> %texel1, float 0.000000e+00, i64 2
  %texel = insertelement <4 x float> %texel2, float 0.000000e+00, i64 3
  call void @air.write_texture_2d.v4f32(ptr addrspace(1) noundef writeonly captures(none) %texture, <2 x i32> %coord, <4 x float> %texel, i32 0, i32 2)
  ret void
}

declare void @air.write_texture_2d.v4f32(ptr addrspace(1) noundef writeonly captures(none), <2 x i32>, <4 x float>, i32, i32) local_unnamed_addr

!air.kernel = !{!0}
!0 = !{ptr @write_texture_2d, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"gid"}
