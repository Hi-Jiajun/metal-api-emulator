; Owned synthetic fixture. Not derived from a third-party metallib.
; Reads one R32Uint texel per invocation at (thread_position_in_grid.x,
; thread_position_in_grid.y) and stores `texel.x + 100` at the cell `4*y + x`.
; A 4x4 R32Uint texture holding 0..15 therefore has to produce [100..115] under
; both dispatch forms conformance/suite-v12.json names: local 4x4 (one group)
; and local 1x1 (16 groups).
;
; The defect this fixture guards against is provider-side: uploading a linear
; image without honouring VkSubresourceLayout.rowPitch leaves every V != 0 texel
; outside the bytes the host wrote, which collapses this expectation to
; [100,101,102,103,100,100,...] under *both* dispatch forms. The thread
; position builtin and the vector coordinate construction are exercised on
; purpose, so a translator-side regression of either stays visible too. The
; single-texel companion in conformance/suite-v11.json reads only (0, 0) and
; cannot see a row stride at all.
source_filename = "kernel_read_texture_2d_cell.metal"

define void @read_texture_2d_cell(ptr addrspace(1) %texture, ptr addrspace(1) %output, <3 x i32> %gid) {
entry:
  %sampler = call ptr addrspace(2) @air.get_read_sampler()
  %x = extractelement <3 x i32> %gid, i32 0
  %y = extractelement <3 x i32> %gid, i32 1
  %coord0 = insertelement <2 x i32> undef, i32 %x, i64 0
  %coord = insertelement <2 x i32> %coord0, i32 %y, i64 1
  %texel = call { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) %texture, ptr addrspace(2) %sampler, <2 x i32> %coord, <2 x i32> zeroinitializer, i32 0, i32 0)
  %value = extractvalue { <4 x i32>, i8 } %texel, 0
  %lane = extractelement <4 x i32> %value, i64 0
  %biased = add i32 %lane, 100
  %linear = shl i32 %y, 2
  %cell = add i32 %linear, %x
  %extent = zext i32 %cell to i64
  %slot = getelementptr inbounds i32, ptr addrspace(1) %output, i64 %extent
  store i32 %biased, ptr addrspace(1) %slot, align 4
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler() local_unnamed_addr

declare { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32) local_unnamed_addr

!air.kernel = !{!0}
!0 = !{ptr @read_texture_2d_cell, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<uint, read>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 64, !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"output"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"gid"}
