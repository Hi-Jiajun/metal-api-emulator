; Owned synthetic fixture. Not derived from a third-party metallib.
source_filename = "kernel_copy_word_with_witness.metal"

define void @copy_word_with_witness(ptr addrspace(1) %input, ptr addrspace(1) %output, ptr addrspace(1) %witness) {
  %value = load i32, ptr addrspace(1) %input, align 4
  %seen = load i32, ptr addrspace(1) %witness, align 4
  store i32 %value, ptr addrspace(1) %output, align 4
  store i32 %seen, ptr addrspace(1) %output, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @copy_word_with_witness, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"input"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"output"}
!5 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"witness"}
