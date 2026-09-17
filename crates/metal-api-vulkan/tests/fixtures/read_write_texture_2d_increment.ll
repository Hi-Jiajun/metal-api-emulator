; Owned synthetic fixture (C2): a read-modify-write kernel over one `R32Float`
; storage image (`texture2d<float, access::read_write>`). Not derived from a
; third-party metallib.
;
; One invocation reads the texel at its own grid coordinate, adds one and writes
; the result back, so a 4x4 view uploaded with `[0.0, 1.0, ..., 15.0]` has to
; read back `[1.0, 2.0, ..., 16.0]`. That reading separates the two halves the
; write-only sibling cannot: the landed bytes depend on the *initial* contents
; the upload path staged (a rail that skipped the upload would read zeroes and
; land a uniform one), and on the storage descriptor being read-write capable
; rather than write-only.
source_filename = "kernel_read_write_texture_2d_increment.metal"

define void @read_write_texture_2d(ptr addrspace(1) captures(none) %texture, <3 x i32> %gid) local_unnamed_addr {
entry:
  %x = extractelement <3 x i32> %gid, i32 0
  %y = extractelement <3 x i32> %gid, i32 1
  %coord0 = insertelement <2 x i32> undef, i32 %x, i64 0
  %coord = insertelement <2 x i32> %coord0, i32 %y, i64 1
  %read = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) captures(none) %texture, <2 x i32> %coord, i32 0, i32 0)
  %texel_read = extractvalue { <4 x float>, i8 } %read, 0
  %lane = extractelement <4 x float> %texel_read, i64 0
  %next = fadd float %lane, 1.000000e+00
  %texel0 = insertelement <4 x float> undef, float %next, i64 0
  %texel1 = insertelement <4 x float> %texel0, float 0.000000e+00, i64 1
  %texel2 = insertelement <4 x float> %texel1, float 0.000000e+00, i64 2
  %texel = insertelement <4 x float> %texel2, float 0.000000e+00, i64 3
  call void @air.write_texture_2d.v4f32(ptr addrspace(1) captures(none) %texture, <2 x i32> %coord, <4 x float> %texel, i32 0, i32 2)
  ret void
}

declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) captures(none), <2 x i32>, i32, i32) local_unnamed_addr

declare void @air.write_texture_2d.v4f32(ptr addrspace(1) captures(none), <2 x i32>, <4 x float>, i32, i32) local_unnamed_addr

!air.kernel = !{!0}
!0 = !{ptr @read_write_texture_2d, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"gid"}
