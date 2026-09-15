; Owned synthetic source counterpart to mrt_declare.metal.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define void @mrt_declare(ptr addrspace(1) %left, ptr addrspace(1) %right, ptr addrspace(1) %output) {
entry:
  %lhs = load i32, ptr addrspace(1) %left, align 4
  %rhs = load i32, ptr addrspace(1) %right, align 4
  %combined = xor i32 %lhs, %rhs
  store i32 %combined, ptr addrspace(1) %output, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @mrt_declare, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"uint*"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"uint*"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint*"}
