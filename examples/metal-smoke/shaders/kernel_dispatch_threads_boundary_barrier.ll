target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define void @kernel_dispatch_threads_boundary_barrier(ptr addrspace(1) %output, <3 x i32> %gid, <3 x i32> %local_size) {
entry:
  tail call void @air.wg.barrier(i32 0, i32 1)
  %x = extractelement <3 x i32> %gid, i64 0
  %y = extractelement <3 x i32> %gid, i64 1
  %row = mul i32 %y, 10
  %index = add i32 %row, %x
  %local_x = extractelement <3 x i32> %local_size, i64 0
  %local_y = extractelement <3 x i32> %local_size, i64 1
  %local_y_scaled = mul i32 %local_y, 100
  %value = add i32 %local_y_scaled, %local_x
  %slot = getelementptr i32, ptr addrspace(1) %output, i32 %index
  store i32 %value, ptr addrspace(1) %slot, align 4
  ret void
}

declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @kernel_dispatch_threads_boundary_barrier, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint*"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3"}
!5 = !{i32 2, !"air.threads_per_threadgroup", !"air.arg_type_name", !"uint3"}
