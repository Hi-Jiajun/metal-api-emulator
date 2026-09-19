; Owned synthetic AIR for the 16-bit shader capability pair (2026-09-20, census
; v48's LPF pipeline: `fixed_frag_lpf_cpf`).
;
; The module carries the census fragment stage's own narrowing shape — a float
; narrowed to `half` by `air.convert`, the half's bits read back through an
; `i16` `lshr` — which is what makes the translator declare
; `OpCapability Float16` beside `OpCapability Int16` (evidence
; `evidence/texture-sampler-d94d8da-2026-09-20/12-pipe58-spirv-capabilities.txt`,
; `6: Shader / 7: Int64 / 8: Int8 / 9: Float16 / 10: Int16`).
;
; The frame is a function of that path rather than a constant: the truncated
; `i16` the module derives — `half(64/255) == 0x3404`, `lshr 1` => `0x1a02` =
; `6658` — is compared against the value the fixture's own arithmetic defines,
; and the stored red channel is the round-tripped half when the two agree and
; `1.0` when they do not. `green` and `blue` are the round-tripped half of
; `128/255` and `192/255`, and `alpha` is the truncated comparison's own
; `select`. So a rail that lost the half conversion, the `i16` shift or the
; comparison lands `ff ...` instead of the attachment's own texel, and a rail
; that lands the texel has executed the whole path.
;
; The three byte/255 texels are exactly the reviewed offscreen fixture's, and
; their half round trip stays inside the same 8-bit step on every driver
; (`half(64/255) == 0.2509765625` rounds to byte `0x40`), so the frame this
; module lands is `40 80 c0 ff` — byte for byte the module the rail executed
; before it could declare these capabilities.
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <4 x float> @render_half_truncated_rgba8() {
entry:
  %r = insertelement <3 x float> undef, float 0.250980406999588, i32 0
  %g = insertelement <3 x float> %r, float 0.501960813999176, i32 1
  %b = insertelement <3 x float> %g, float 0.7529411911964417, i32 2
  %h = fptrunc <3 x float> %b to <3 x half>
  %bits = bitcast <3 x half> %h to <3 x i16>
  %trunc = lshr <3 x i16> %bits, splat (i16 1)
  %low = extractelement <3 x i16> %trunc, i64 0
  %index = zext i16 %low to i64
  %exact = icmp eq i64 %index, 6658
  %r0 = extractelement <3 x half> %h, i64 0
  %g1 = extractelement <3 x half> %h, i64 1
  %b2 = extractelement <3 x half> %h, i64 2
  %rf = fpext half %r0 to float
  %gf = fpext half %g1 to float
  %bf = fpext half %b2 to float
  %red = select i1 %exact, float %rf, float 1.000000e+00
  %alpha = select i1 %exact, float 1.000000e+00, float 0.000000e+00
  %out0 = insertelement <4 x float> undef, float %red, i32 0
  %out1 = insertelement <4 x float> %out0, float %gf, i32 1
  %out2 = insertelement <4 x float> %out1, float %bf, i32 2
  %out3 = insertelement <4 x float> %out2, float %alpha, i32 3
  ret <4 x float> %out3
}

!air.fragment = !{!0}
!0 = !{ptr @render_half_truncated_rgba8, !1, !2}
!1 = !{!3}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{}
