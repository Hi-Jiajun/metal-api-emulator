; Owned synthetic AIR for the *translated* stage-buffer shape (R9c): the same
; 2x2 offscreen fragment stage as `render_offscreen_2x2.frag.ll`, with its
; colour read from the stage's own `[[buffer(0)]]` argument instead of being an
; immediate.
;
; `render_offscreen_2x2_buffered.frag.ll` is the same declaration in the
; counterexample's role: its translation is refused at registration because the
; contract declares no slot. This fixture is the executable twin of that one —
; the contract does declare the slot
; (`StageBufferBinding { stage: Fragment, index: 0, access: Read,
; footprint: Static { max_bytes: 16 } }`), the pass binds its bytes
; (`StageBufferView`), and the descriptor slot the module reads is the one the
; reflection names (the translator's default layout puts `[[buffer(n)]]` at
; `DescriptorSet 0 / Binding n`), so the attachment's readback is the buffer's
; own payload through the format's quantisation: `(64/255, 128/255, 192/255,
; 1)` lands as `40 80 c0 ff`.
;
; The load is one 16-byte `float4`, which is the static footprint the contract
; declares; a stage whose reach the declaration cannot cover is refused at
; registration rather than executed.
;
; Translated by the provider's own `TranslatedRenderStage::translate` in the
; tests, exactly as the other `.ll` fixtures in this directory are — the
; toolchain is the one the translator itself drives, so no committed binary is
; needed here.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <4 x float> @render_stage_buffer_rgba8(ptr addrspace(1) %tint) {
entry:
  %value = load <4 x float>, ptr addrspace(1) %tint, align 16
  ret <4 x float> %value
}

!air.fragment = !{!0}
!0 = !{ptr @render_stage_buffer_rgba8, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!4}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"tint"}
