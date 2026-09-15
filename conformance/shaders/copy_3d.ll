; Owned synthetic source counterpart to copy_3d.metal.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define void @copy_3d(ptr addrspace(1) %input, ptr addrspace(1) %output, <3 x i32> %gid) {
entry:
  %x = extractelement <3 x i32> %gid, i32 0
  %y = extractelement <3 x i32> %gid, i32 1
  %z = extractelement <3 x i32> %gid, i32 2
  %plane = mul i32 %z, 3
  %row = add i32 %plane, %y
  %base = mul i32 %row, 5
  %index = add i32 %base, %x
  %src = getelementptr i32, ptr addrspace(1) %input, i32 %index
  %value = load i32, ptr addrspace(1) %src, align 4
  %dst = getelementptr i32, ptr addrspace(1) %output, i32 %index
  store i32 %value, ptr addrspace(1) %dst, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @copy_3d, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"uint*"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 9, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint*"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3"}
