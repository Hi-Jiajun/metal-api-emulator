; Owned synthetic AIR for the *writable* stage-buffer shape (R9f): the 2x2
; offscreen fragment stage with two `[[buffer(N)]]` arguments — a read-only
; source and a writable sink.
;
; The stage stores the source's own payload into the sink before returning it
; as the colour output, so one execution has two falsifiable readings: the
; covered texel is the source's payload through the format's quantisation, and
; the sink's bytes after the pass are that same payload. A rail that binds the
; sink read-only, swaps the two slots, or drops the writeback leaves one of the
; two unchanged when the source's payload moves.
;
; The write is a whole-`float4` store, which is the one shape the contract's
; write pairing admits in this increment: the reflection classifies the sink as
; `air.write` and the contract has to declare the same access, because a
; declaration that claims a write the module does not perform (or the reverse)
; is refused by name at registration.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <4 x float> @render_stage_buffer_write_rgba8(ptr addrspace(1) %source, ptr addrspace(1) %sink) {
entry:
  %value = load <4 x float>, ptr addrspace(1) %source, align 16
  store <4 x float> %value, ptr addrspace(1) %sink, align 16
  ret <4 x float> %value
}

!air.fragment = !{!0}
!0 = !{ptr @render_stage_buffer_write_rgba8, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4, !5}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"source"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"sink"}
