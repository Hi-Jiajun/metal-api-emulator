; Owned synthetic fixture. Not derived from a third-party metallib.
; Reads one R32Uint texel from a texture2d at the thread position in the grid.
source_filename = "kernel_read_texture_2d.metal"

; The thread position is the vector form every passing fixture uses; the
; scalar `uint` form the translator accepts leaves later invocations with a
; zero thread id, which the multi-invocation regression test pins.
define void @read_texture_2d(ptr addrspace(1) %texture, ptr addrspace(1) %output, <3 x i32> %gid) {
entry:
  ; The reviewed multi-invocation fixtures all begin with a workgroup barrier;
  ; it is what makes the translator materialize per-invocation positions
  ; rather than folding them to zero.
  tail call void @air.wg.barrier(i32 0, i32 1)
  %sampler = call ptr addrspace(2) @air.get_read_sampler()
  %tid = extractelement <3 x i32> %gid, i32 0
  %row = extractelement <3 x i32> %gid, i32 1
  %coord.x = and i32 %tid, 1
  %coord.y = and i32 %row, 3
  %coord0 = insertelement <2 x i32> undef, i32 %coord.x, i64 0
  %coord = insertelement <2 x i32> %coord0, i32 %coord.y, i64 1
  %texel = call { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) %texture, ptr addrspace(2) %sampler, <2 x i32> %coord, <2 x i32> zeroinitializer, i32 0, i32 0)
  %value = extractvalue { <4 x i32>, i8 } %texel, 0
  %lane = extractelement <4 x i32> %value, i64 0
  ; One output word per (column, row) cell of the 4x4 grid.
  %linear = shl i32 %row, 2
  %cell = add i32 %linear, %tid
  %extent = zext i32 %cell to i64
  %slot = getelementptr inbounds i32, ptr addrspace(1) %output, i64 %extent
  store i32 %lane, ptr addrspace(1) %slot, align 4
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler() local_unnamed_addr

declare void @air.wg.barrier(i32, i32)

declare { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32) local_unnamed_addr

!air.kernel = !{!0}
!0 = !{ptr @read_texture_2d, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<uint, read>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 64, !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"output"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"gid"}
