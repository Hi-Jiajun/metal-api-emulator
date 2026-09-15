; Owned synthetic source counterpart to mrt_declare4.metal.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define void @mrt_declare4(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %output) {
entry:
  %lhs = load i32, ptr addrspace(1) %a, align 4
  %mid = load i32, ptr addrspace(1) %b, align 4
  %mid2 = load i32, ptr addrspace(1) %c, align 4
  %rhs = load i32, ptr addrspace(1) %d, align 4
  %left = xor i32 %lhs, %mid
  %right = xor i32 %mid2, %rhs
  %combined = xor i32 %left, %right
  store i32 %combined, ptr addrspace(1) %output, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @mrt_declare4, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"uint*"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"uint*"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"uint*"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"uint*"}
!7 = !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint*"}
