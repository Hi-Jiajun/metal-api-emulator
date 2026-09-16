// Capture native Metal observations for the bounded compute-buffer-v1 through v13 suites.
// Build on macOS with Swift 5 language mode and link Foundation, Metal,
// CoreGraphics, and CryptoKit. This file does not implement ComputeProvider.
//
// `conformance/suite-v13.json` is the first committed suite that declares
// `render_cases`, and `conformance/compare.py` now has the matching attachment
// section, so a render case reports through the same writebacks/allocations
// shape the compute cases use. `capture` runs a suite's render cases after its
// compute cases, so `--suite conformance/suite-v13.json` is the path that puts
// the attachment comparison into the capture matrix; `--render-selftest`
// remains the suite-free path to the same reviewed fixture, and it is the one
// observation an Apple GPU has produced so far (`conformance/RENDER-CAPTURE.md`
// §5).
import Foundation
import Metal
import CoreGraphics
import CryptoKit
import Dispatch
import Darwin

private let maximumFileBytes = 1_048_576
private let maximumAllocationBytes: UInt64 = 1_048_576
private let maximumPassCount = 8

private struct OracleError: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}

private func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw OracleError(message) }
}

private struct SourceDefinition: Decodable, Equatable {
    let path: String
    let sha256: String
}

private struct BufferDefinition: Decodable {
    let binding: UInt64
    let allocation: UInt64
    let view: UInt64
    let offset: UInt64
    let length: UInt64
    let allocation_size: UInt64
    let access: String
    let initial_hex: String
}

private struct Writeback: Codable {
    let allocation: UInt64
    let view: UInt64
    let offset: UInt64
    let bytes_hex: String
}

private struct DispatchDefinition: Decodable, Equatable {
    let grid: [UInt64]
    let local: [UInt64]
    let bindings: [UInt64]?
    let program: Int?
}

private struct BufferSlotDefinition: Decodable, Equatable {
    let binding: UInt64
    let access: String
    let length: UInt64
}

private struct ProgramDefinition: Decodable, Equatable {
    let entry: String
    let air: SourceDefinition
    let metal: SourceDefinition
    let buffer_slots: [BufferSlotDefinition]?
}

private struct CaseDefinition: Decodable {
    let id: String
    let entry: String
    let grid: [UInt64]
    let local: [UInt64]
    let dispatches: [DispatchDefinition]?
    let programs: [ProgramDefinition]?
    let command_buffers: [[Int]]?
    let air: SourceDefinition
    let metal: SourceDefinition
    let buffers: [BufferDefinition]
    let textures: [TextureDefinition]?
    let expected_writebacks: [Writeback]
    /// Which capture rails the suite marks this compute case executable on.
    /// `nil` keeps the pre-v15 shape: a case every rail owes. A case that
    /// carries a heap or indirect section is only executable on the rails its
    /// marker names (`research/docs/25` §5.2), and this oracle is not one of
    /// them yet.
    let capture_rails: [String]?
}

private struct TextureDefinition: Decodable {
    let binding: Int
    let allocation: UInt64
    let view: UInt64
    let width: Int
    let height: Int
    let format: String
    let access: String
    let initial_hex: String
}

/// The reviewed render fixture's source identity.
///
/// Deliberately a distinct type from `SourceDefinition`: a render pipeline is
/// one module carrying two stage entries, so the review pins the pair. The pin
/// is still code-side (`reviewedRenderModule`), because re-hashing a fixture
/// must not be enough to admit a different module for execution.
private struct RenderSourcePin: Decodable, Equatable {
    let path: String
    let sha256: String
}

/// The colour attachment a render case draws into.
///
/// The fields mirror `metal_api_core::provider::RenderAttachment`: identity,
/// format, extent and the load/store pair. `clear_hex` is the four-byte clear
/// value the contract carries in memory order, and `initial_hex` is what a
/// `load` case keeps from its previous contents.
private struct RenderAttachmentDefinition: Decodable {
    let allocation: UInt64
    let view: UInt64
    let format: String
    let width: Int
    let height: Int
    let load: String
    let store: String
    let clear_hex: String?
    let initial_hex: String?
    /// The MRT case's per-attachment expectation; absent for the
    /// single-attachment form, whose expectation is case-level.
    let expected_hex: String?
}

/// The depth attachment a render case declares (`research/docs/23` §3.3,
/// v36; the store pair is v43).
///
/// The fields mirror `metal_api_core::provider::RenderDepthAttachment`: the
/// format spelling, the extent and the load operation with the depth a clear
/// starts from. The surface is rail-owned — no trace identity and no readback
/// — so the default shape names no view for it, and the three fields below
/// stay absent with the store action (fail-closed).
///
/// A case that stores the depth surface (`research/docs/23` §3.3, v43) states
/// all four of them: the store action, the identity its texels land in and the
/// bytes the readback has to show there. The depth texels travel through the
/// same writeback channel a colour attachment uses.
private struct DepthAttachmentDefinition: Decodable {
    let format: String
    let width: Int
    let height: Int
    let load: String
    /// The depth a `"clear"` load starts from; absent for any other load.
    let clear_depth: Double?
    /// The pass's depth store action (`research/docs/23` §3.3, v43). The only
    /// spelling this increment admits is `"store"`, which is what makes the
    /// texels observable; a surface the pass discards is the *absent* field
    /// rather than a second spelling, so no `"dontcare"` arm exists here.
    let store: String?
    /// The allocation the stored depth texels land in, present exactly when
    /// `store` is.
    let allocation: UInt64?
    /// The view inside that allocation the landing covers, present exactly
    /// when `store` is.
    let view: UInt64?
    /// The reviewed depth texels, present exactly when `store` is: an image in
    /// the surface's own `depth32float` texel layout, compared byte for byte
    /// against the readback like a colour attachment's expectation. It has to
    /// differ from the clear image, or "the pass stored the texels" and "it
    /// never wrote depth" would read the same.
    let expected_hex: String?
}

/// The depth state a render case declares (`research/docs/23` §3.3, v36).
///
/// The fields mirror `metal_api_core::provider::DepthTest`: the compare
/// function spelling and whether fragments write depth. Metal states both on
/// one `MTLDepthStencilDescriptor` per encoder.
private struct DepthTestDefinition: Decodable {
    let compare: String
    let write: Bool
}

/// The culling state a render case declares (`research/docs/23` §3.3, v39).
///
/// The fields mirror `metal_api_core::provider::RenderPassCull`: the refusal
/// mode (`"none"`, `"front"`, `"back"`) and the winding that names a front face
/// (`"clockwise"`, `"counter_clockwise"`). Metal states both on the encoder
/// (`setCullMode` / `setFrontFacingWinding`), and a case that declares nothing
/// keeps the default every earlier case has: no culling.
private struct CullDefinition: Decodable {
    let mode: String
    let winding: String
}

/// The blend state a render case declares (`research/docs/23` §3.3, v40).
///
/// The fields mirror `metal_api_core::provider::BlendAttachment`: the source
/// and destination factors of the RGB and of the alpha channel, plus the
/// operation that combines them. Each factor is a wire spelling (`"zero"`,
/// `"one"`, `"source_alpha"`, `"one_minus_source_alpha"`) and the operation is
/// `"add"`. Metal states all five on one pipeline colour attachment
/// (`sourceRGBBlendFactor`, `destinationAlphaBlendFactor`, …) rather than on
/// the encoder, and a case that declares nothing keeps the semantics every
/// earlier case has: the fragment output lands unblended.
private struct BlendAttachmentDefinition: Decodable {
    let source_rgb: String
    let destination_rgb: String
    let source_alpha: String
    let destination_alpha: String
    let operation: String
}

/// The pass-wide multisample state a render case declares
/// (`research/docs/23` §3.3, v51).
///
/// The reviewed shape is the four-sample raster both rails spell
/// `MTLSampleCount4`/`TYPE_4`: Metal renders into a four-sample texture with
/// `rasterSampleCount = 4` and resolves it into the attachment's own texture
/// with `storeAction = .multisampleResolve`, which is what the oracle reads
/// back. The state is the pass's own, so it carries no attachment identity.
private struct MultisampleDefinition: Decodable {
    let sample_count: UInt64
}

/// The depth resolve one render case states (`research/docs/23` §3.3, v57).
///
/// The pass-level filter the two APIs spell differently: the case's own
/// spelling is the closed family's wire name (`"sample0"`/`"min"`/`"max"`),
/// and the rails map it onto their own constants. The reviewed fixture states
/// `"sample0"`, the filter the native rail declares; the device-gated pair
/// states `"min"`/`"max"` beside a gate of the same name
/// (`research/docs/23` §3.3, v57d).
private struct DepthResolveDefinition: Decodable {
    let filter: String
}

/// The stencil resolve one render case states (`research/docs/23` §3.3, v60).
///
/// The pass-level filter Metal spells as a two-value family: `"sample0"` takes
/// sample zero and `"depth_resolved_sample"` takes the sample the depth
/// resolve selected. The reviewed pair states one of each beside the
/// combined depth-stencil surface.
private struct StencilResolveDefinition: Decodable {
    let filter: String
}

/// The stencil attachment a render case declares (`research/docs/23` §3.3,
/// v47; the store pair is v49).
///
/// The fields mirror `metal_api_core::provider::RenderStencilAttachment`: the
/// format spelling, the extent and the load operation with the value a clear
/// starts from. The surface is rail-owned — no trace identity and no readback
/// — so the default shape names no view for it, and the four fields below stay
/// absent with the store action (fail-closed).
///
/// A case that stores the stencil surface (`research/docs/23` §3.3, v49)
/// states all four of them: the store action, the identity its texels land in
/// and the bytes the readback has to show there. One `stencil8` texel is one
/// byte, so the landing is `width * height` bytes, and it travels through the
/// same writeback channel a colour or depth attachment uses.
private struct StencilAttachmentDefinition: Decodable {
    let format: String
    let width: Int
    let height: Int
    let load: String
    /// The value a `"clear"` load starts from; absent for any other load.
    let clear_value: UInt32?
    /// The pass's stencil store action (`research/docs/23` §3.3, v49). The
    /// only spelling this increment admits is `"store"`, which is what makes
    /// the texels observable; a surface the pass discards is the *absent*
    /// field rather than a second spelling, so no `"dontcare"` arm exists
    /// here.
    let store: String?
    /// The allocation the stored stencil texels land in, present exactly when
    /// `store` is.
    let allocation: UInt64?
    /// The view inside that allocation the landing covers, present exactly
    /// when `store` is.
    let view: UInt64?
    /// The reviewed stencil texels, present exactly when `store` is: an image
    /// in the surface's own one-byte `stencil8` texel layout, compared byte
    /// for byte against the readback like a colour attachment's expectation.
    /// It has to differ from the clear image, or "the pass stored the texels"
    /// and "it never wrote stencil" would read the same.
    let expected_hex: String?
}

/// The stencil state a render case declares (`research/docs/23` §3.3, v47).
///
/// The fields mirror `metal_api_core::provider::StencilTest`: the compare
/// function spelling, the reference value the test compares against, the two
/// byte-wide masks, and the three operations — what a fragment that fails the
/// stencil test does, what one that passes it but fails the depth test does,
/// and what one that passes both does. Metal states them on one
/// `MTLStencilDescriptor` per face of a `MTLDepthStencilDescriptor`, with the
/// reference value on the encoder rather than on the pipeline.
private struct StencilTestDefinition: Decodable {
    let compare: String
    let reference: UInt32
    let read_mask: UInt32
    let write_mask: UInt32
    let fail_op: String
    let depth_fail_op: String
    let pass_op: String
}

/// One attribute of a render case's vertex stream.
///
/// The fields mirror `metal_api_core::provider::VertexAttribute`: the
/// shader-visible location, the byte offset inside one vertex and the format the
/// rails translate. The format is a string because it is a wire spelling
/// (`"float32x2"`, …), not a `MTLVertexFormat`.
private struct RenderVertexAttributeDefinition: Decodable, Equatable {
    let location: UInt64
    let offset: UInt64
    let format: String
}

/// One vertex stream of a render case's layout
/// (`metal_api_core::provider::VertexBufferLayout`).
///
/// `step` is the v31 stream advance (`research/docs/23` §3.3): `"per_vertex"`
/// or `"per_instance"`. A fixture that says nothing about it describes the
/// per-vertex stream every pre-v31 layout meant — the same default
/// `metal_api_core::provider`'s `default_vertex_step` applies — so equality
/// resolves the missing field and the explicit spelling to one meaning, and
/// the reviewed pins below can state `"per_vertex"` while a pre-v31 fixture
/// stays silent.
private struct RenderVertexBufferLayoutDefinition: Decodable, Equatable {
    let stride: UInt64
    let step: String?
    let attributes: [RenderVertexAttributeDefinition]

    /// The stream's advance with the pre-v31 default applied.
    var resolvedStep: String { step ?? "per_vertex" }

    static func == (lhs: RenderVertexBufferLayoutDefinition,
                    rhs: RenderVertexBufferLayoutDefinition) -> Bool {
        lhs.stride == rhs.stride
            && lhs.attributes == rhs.attributes
            && lhs.resolvedStep == rhs.resolvedStep
    }
}

/// The vertex-input shape a render case draws with: one entry per bound stream,
/// in binding order (`metal_api_core::provider::VertexLayout::Buffers`).
private struct RenderVertexLayoutDefinition: Decodable, Equatable {
    let buffers: [RenderVertexBufferLayoutDefinition]
}

/// One vertex stream bound for a render case: the view whose bytes the draw
/// reads, in binding order (`metal_api_core::provider::BufferView`,
/// `research/docs/23` §3.6).
///
/// The position in the array is the binding index, and the bytes travel with the
/// view — `initial_hex` is exactly `length` bytes — so no compute case has to
/// declare the stream. `offset` is where the view starts inside its allocation,
/// which is where the oracle places its binding.
private struct RenderVertexBufferDefinition: Decodable {
    let allocation: UInt64
    let view: UInt64
    let offset: UInt64
    let length: UInt64
    let initial_hex: String
}

/// The index buffer a render case draws through: the same view fields plus the
/// width of the indices (`metal_api_core::provider::IndexBufferBinding`).
private struct RenderIndexBufferDefinition: Decodable {
    let allocation: UInt64
    let view: UInt64
    let offset: UInt64
    let length: UInt64
    let initial_hex: String
    let format: String
}

/// One offscreen render case (`research/docs/23` §1.2, §5.1).
private struct RenderCaseDefinition: Decodable {
    let id: String
    /// The compute case whose pass declares the attachment view. The oracle
    /// validates the render case's own shape, and the declaring case's
    /// whole-allocation read view is the exception `validateBuffers` grants.
    let declaring_case: String
    let vertex_entry: String
    let fragment_entry: String
    let metal: RenderSourcePin
    let vertices: UInt64
    let viewport: [UInt64]
    /// The pass's scissor rectangle in framebuffer coordinates, or `nil` for the
    /// whole attachment (`research/docs/23` §3.3, v29).
    let scissor: [UInt64]?
    /// How many instances the pass's single draw runs, or `nil` for one
    /// instance — the shape every pre-v31 case declares
    /// (`metal_api_core::provider::RenderPassDescriptor::instance_count`,
    /// `research/docs/23` §3.3, v31). The reviewed instanced case declares two.
    let instance_count: UInt64?
    /// The vertex offset every index value of the draw is read through, or
    /// `nil` for zero — the semantics every pre-v34 case has
    /// (`metal_api_core::provider::RenderPassDescriptor::base_vertex`,
    /// `research/docs/23` §3.3, v34). Metal's `baseVertex` and Vulkan's
    /// `vertexOffset` only exist for an indexed draw, so a case that declares
    /// one without an index buffer is refused. The reviewed base-vertex case
    /// declares one over its five-vertex stream.
    let base_vertex: UInt64?
    /// The vertex-input half, absent for the `vertex_id` shape
    /// (`research/docs/23` §3.3). A case that carries a layout draws the
    /// indexed reviewed module instead: the layout, its bindings and the index
    /// buffer arrive together or not at all, and each binding carries its own
    /// bytes (`research/docs/23` §3.6).
    let vertex_layout: RenderVertexLayoutDefinition?
    let vertex_buffers: [RenderVertexBufferDefinition]?
    let indices: RenderIndexBufferDefinition?
    /// The first render increments' single attachment, or `nil` for an MRT
    /// case that declares `attachments` instead. Exactly one of the two
    /// fields is present.
    let attachment: RenderAttachmentDefinition?
    /// The MRT case's attachment list, in location order; mutually exclusive
    /// with `attachment`.
    let attachments: [RenderAttachmentDefinition]?
    /// The single-attachment case's expectation. An MRT case leaves this
    /// absent and spells the expectation on each attachment entry instead.
    let expected_hex: String?
    /// The coverage claim (`research/docs/23` §3.3, v38): `"partial"` says the
    /// pass's single draw covers only part of the attachment, so the
    /// expectation states the texels the draw missed as the colour the clear
    /// load started them from. An absent claim keeps the milestone's rule
    /// since v13: a clearing pass claims every texel.
    let coverage: String?
    /// The pass-wide multisample raster (`research/docs/23` §3.3, v51), or
    /// `nil` for the single-sample raster every pre-v51 case runs. The reviewed
    /// case states four samples and observes the resolve of the fragment output
    /// and the load's own colour in the attachment view: its expectation is a
    /// k-of-four mix of the two, which a single-sample raster cannot produce.
    let multisample: MultisampleDefinition?
    /// The depth resolve a stored multisampled depth surface states
    /// (`research/docs/23` §3.3, v57c), or `nil` for a pass that resolves
    /// nothing. Only legal beside a multisample raster whose depth attachment
    /// is stored; the reviewed fixture states the `sample0` filter and its
    /// marker names this oracle's rail, so this rail maps the filter onto the
    /// native `MTLMultisampleDepthResolveFilter` and reads the landing back.
    let depth_resolve: DepthResolveDefinition?
    /// The device gate one depth-resolve case may state (`research/docs/23`
    /// §3.3, v57d): the case appears in a capture if and only if the device's
    /// declared mask carries the named filter's bit. The marker still decides
    /// which rails own the case; the gate is the device-side half of the same
    /// question. The v57e self-test proved Apple Paravirtual executes Min and
    /// Max, so this oracle's mask carries all three bits and a Min/Max-gated
    /// case the marker names is present in the oracle capture.
    let requires_depth_resolve_filter: String?
    /// The stencil resolve a stored multisampled stencil surface states
    /// (`research/docs/23` §3.3, v60), or `nil` for a pass that resolves
    /// nothing. Only legal beside a multisample raster whose stencil
    /// attachment is stored; the `depth_resolved_sample` filter additionally
    /// names the sample the depth resolve selected.
    let stencil_resolve: StencilResolveDefinition?
    /// The device gate one stencil-resolve case may state
    /// (`research/docs/23` §3.3, v60): the case appears in a capture if and
    /// only if the device's declared mask carries the named filter's bit. The
    /// v59 self-test proved Apple Paravirtual executes both filters, so this
    /// oracle's mask carries both bits.
    let requires_stencil_resolve_filter: String?
    /// The wildcard channel (`research/docs/23` §3.3, v33): the row-major
    /// texel indices of the single attachment whose bytes the case does *not*
    /// claim, stated in advance. Only a `dontcare` load may leave texels
    /// unclaimed, because the undefined pre-pass contents are exactly what
    /// makes an unclaimed byte legitimate. An absent list means the case
    /// claims every texel, the semantics every case before v33 has.
    let wildcard_texels: [Int]?
    /// The rail-owned depth attachment the pass opens, or `nil` for no depth
    /// surface — the shape every case before v36 declares
    /// (`metal_api_core::provider::RenderPassDescriptor::depth`,
    /// `research/docs/23` §3.3, v36). The reviewed depth case clears one
    /// `depth32float` surface to one, and the surface covers the same render
    /// area as the colour attachment.
    let depth: DepthAttachmentDefinition?
    /// The pass's depth state, or `nil` for no depth state — the shape every
    /// pre-v36 case declares. A state only exists for a pass that opens a
    /// depth attachment, and the reviewed depth case states a `less` test with
    /// writes on (`research/docs/23` §3.3, v36).
    let depth_test: DepthTestDefinition?
    /// The pass's culling state, or `nil` for no culling — the shape every
    /// pre-v39 case declares (`metal_api_core::provider::RenderPassDescriptor::cull`,
    /// `research/docs/23` §3.3, v39). The reviewed cull case culls back faces
    /// with a counter-clockwise front, so exactly the counter-clockwise copy of
    /// its two opposite-order triangles survives.
    let cull: CullDefinition?
    /// The pass's blend state, or `nil` for the fragment output landing
    /// unblended — the semantics every case before v40 has
    /// (`metal_api_core::provider::RenderPassDescriptor::blend`,
    /// `research/docs/23` §3.3, v40). Metal states the state on the colour
    /// attachment of the pipeline descriptor rather than on the encoder, and
    /// the reviewed blend case blends source alpha against
    /// one-minus-source-alpha with an add over a cleared-to-zero attachment,
    /// so a rail that ignored the state would store the tint itself.
    let blend: [BlendAttachmentDefinition]?
    /// The rail-owned stencil attachment the pass opens, or `nil` for no
    /// stencil surface — the shape every case before v47 declares
    /// (`metal_api_core::provider::RenderStencilAttachment`,
    /// `research/docs/23` §3.3, v47). The reviewed stencil case clears one
    /// `stencil8` surface to zero, covering the same render area as the colour
    /// attachment, and the values the pass stores are what decide which half
    /// of its draw survives. The surface is rail-owned where the case declares
    /// no store action, and the observation stays the colour attachment there;
    /// the v49 store pair (`§3.3`, v49) adds the identity its texels land in
    /// and turns the surface itself into an observed half.
    let stencil: StencilAttachmentDefinition?
    /// The pass's stencil state, or `nil` for no stencil state — the shape
    /// every pre-v47 case declares. A state only exists for a pass that opens
    /// a stencil attachment, and the reviewed stencil case states the `equal`
    /// test against reference zero that writes the value the next primitive of
    /// the same draw is then tested against (`research/docs/23` §3.3, v47).
    let stencil_test: StencilTestDefinition?
    /// Which capture rails the suite marks this render case executable on. The
    /// oracle validates every render case's metadata, but it only *runs* the
    /// ones its marker names (`conformance/compare.py` refuses a rail that
    /// reports a case its marker does not name). v14's present case is marked
    /// for the provider rails; its Apple evidence is `--present-selftest`.
    let capture_rails: [String]
}

/// A render case whose shape, source identity and expectation are reviewed.
private struct ValidatedRender {
    let definition: RenderCaseDefinition
    let source: String
    /// One entry per colour attachment, in location order.
    let attachments: [ValidatedRenderAttachment]
    /// One entry per bound vertex stream, in binding order, with the bytes the
    /// case's own views carry. Empty for the `vertex_id` shape.
    let vertexStreams: [ValidatedVertexStream]
    /// The index buffer of an indexed case, with its footprint and index values
    /// already proved against the streams above.
    let indexStream: ValidatedIndexStream?
    /// The reviewed depth surface and state, or `nil` for the depth-less shape
    /// every pre-v36 case declares (`research/docs/23` §3.3, v36). The runner
    /// states these on the pass descriptor and the encoder.
    let depth: ValidatedDepth?
    /// The reviewed stencil surface and state, or `nil` for the stencil-less
    /// shape every pre-v47 case declares (`research/docs/23` §3.3, v47). The
    /// runner states these on the pass descriptor and the encoder. The surface
    /// itself is rail-owned unless the case states the v49 store pair, which is
    /// what adds the landing and the readback below.
    let stencil: ValidatedStencil?
}

/// One vertex stream the draw reads: its binding index, stride, advance,
/// attributes and the bytes themselves, taken from the view the case declares.
private struct ValidatedVertexStream {
    let binding: Int
    let stride: UInt64
    /// The reviewed stream advance, resolved to `"per_vertex"` /
    /// `"per_instance"` (`research/docs/23` §3.3, v31): both the descriptor's
    /// `stepFunction` and the footprint proof read it.
    let step: String
    let attributes: [RenderVertexAttributeDefinition]
    /// Where the view starts inside its allocation, which is where the binding
    /// points (`render.rs::stream_buffer`).
    let offset: UInt64
    let bytes: Data
}

/// The two index widths the contract admits, with the Metal type and the byte
/// width the footprint proof reads (`metal_api_core::provider::IndexFormat`).
private enum ReviewedIndexType {
    case uint16
    case uint32

    init?(spelling: String) {
        switch spelling {
        case "uint16": self = .uint16
        case "uint32": self = .uint32
        default: return nil
        }
    }

    var metal: MTLIndexType {
        self == .uint16 ? .uint16 : .uint32
    }

    var byteWidth: UInt64 {
        self == .uint16 ? 2 : 4
    }
}

/// The index buffer of an indexed case, with the count and the span the streams
/// have to cover.
private struct ValidatedIndexStream {
    let format: ReviewedIndexType
    let indexCount: UInt64
    /// Where the view starts inside its allocation, for the draw call's
    /// `indexBufferOffset`.
    let offset: UInt64
    /// The draw's vertex offset (`research/docs/23` §3.3, v34): the footprint
    /// proofs read the indices *after* it, and the draw call carries it as
    /// Metal's `baseVertex`.
    let baseVertex: UInt64
    let bytes: Data
}

private struct ValidatedRenderAttachment {
    let allocation: UInt64
    let view: UInt64
    let width: Int
    let height: Int
    let load: String
    /// The contract's store operation: `"store"` keeps the attachment on the
    /// observable surface, `"dontcare"` discards it (`research/docs/23` §3.6,
    /// v19).
    let store: String
    let clearComponents: [Double]
    let initial: Data?
    /// The reviewed expectation of a stored attachment; `nil` for a discarded
    /// attachment, which carries no expectation and no observation.
    let expected: Data?
    /// The byte offsets inside this attachment's own texel image that the case
    /// does not claim (`research/docs/23` §3.3, v33): the four bytes of every
    /// texel the wildcard list names. The readback comparison steps over them
    /// and the reported bytes stay the measured ones; the empty set is every
    /// attachment that declares no wildcard list.
    let wildcardBytes: Set<Int>
    /// The ``MTLPixelFormat`` the case's declared attachment format names
    /// (`research/docs/23` §3.3, v21): the texture and the pipeline attachment
    /// both take it, so the case's expected texels pin which channel order the
    /// attachment is observing.
    let pixelFormat: MTLPixelFormat
}

/// The reviewed depth pair a case carries (`research/docs/23` §3.3, v36): the
/// `depth32float` surface's extent, the depth its clear starts from and the
/// `less` test with writes on. The review above already forced these to the
/// reviewed values, so the runner only has to state them on the pass
/// descriptor and on the encoder's depth-stencil state. The optional store
/// pair (`§3.3`, v43) says whether the texels outlive the pass and where they
/// land.
private struct ValidatedDepth {
    let width: Int
    let height: Int
    let clearDepth: Double
    let isLess: Bool
    let write: Bool
    /// The stored surface's landing and expectation, or `nil` for the
    /// rail-owned shape every pre-v43 case declares: the pass discards the
    /// texels exactly as `render.rs::depth_texture` does, so there is no
    /// readback, no writeback and no allocation observation.
    let store: ValidatedDepthStore?
}

/// The landing a stored depth surface names (`research/docs/23` §3.3, v43):
/// the allocation and view the depth texels land in at offset zero, and the
/// bytes the readback has to show there. The observation travels through the
/// same writeback / allocation channel a colour attachment uses.
private struct ValidatedDepthStore {
    let allocation: UInt64
    let view: UInt64
    let expected: Data
}

/// The reviewed stencil surface and state a case carries
/// (`research/docs/23` §3.3, v47): the `stencil8` surface's extent, the value
/// its clear starts from and the reference the test compares against. The
/// review above already forced these to the reviewed values, so the runner
/// only has to state the surface on the pass descriptor and the test on the
/// encoder's depth-stencil state. The optional store pair (`§3.3`, v49) says
/// whether the texels outlive the pass and where they land.
private struct ValidatedStencil {
    let width: Int
    let height: Int
    let clearValue: UInt32
    let reference: UInt32
    /// The stored surface's landing and expectation, or `nil` for the
    /// rail-owned shape every pre-v49 case declares: the pass discards the
    /// texels exactly as `render.rs::stencil_texture` does, so there is no
    /// readback, no writeback and no allocation observation.
    let store: ValidatedStencilStore?
}

/// The landing a stored stencil surface names (`research/docs/23` §3.3, v49):
/// the allocation and view the stencil texels land in at offset zero, and the
/// bytes the readback has to show there — one byte per texel, where the depth
/// sibling's landing carries four. The observation travels through the same
/// writeback / allocation channel a colour attachment uses.
private struct ValidatedStencilStore {
    let allocation: UInt64
    let view: UInt64
    let expected: Data
}

private struct SuiteDefinition: Decodable {
    let schema_version: UInt64
    let suite: String
    let guard_byte: UInt8
    let cases: [CaseDefinition]
    let render_cases: [RenderCaseDefinition]?
}

private struct ValidatedBuffer {
    let definition: BufferDefinition
    let backing: Data
}

private struct ValidatedCase {
    let definition: CaseDefinition
    let dispatches: [DispatchDefinition]
    let commandBuffers: [[Int]]
    let programs: [(definition: ProgramDefinition, source: String)]
    let buffers: [ValidatedBuffer]
    let textures: [ValidatedTexture]
}

private struct ValidatedTexture {
    let definition: TextureDefinition
    let backing: Data
}

private struct ValidatedSuite {
    let name: String
    let sha256: String
    let cases: [ValidatedCase]
    let renderCases: [ValidatedRender]
}

private struct AllocationResult: Encodable {
    let allocation: UInt64
    let bytes_hex: String
}

private struct CaseResult: Encodable {
    let id: String
    let completion: String
    let writebacks: [Writeback]
    let allocations: [AllocationResult]
}

/// The depth resolve capability mask this oracle declares
/// (`research/docs/23` §3.3, v57c/v57d/v57f): bit `i` is the filter whose wire
/// code is `i`. The v57e `--depth-resolve-selftest` run measured the Apple
/// Paravirtual device executing all three filters — the mixed column lands
/// `min=0000003f` and `max=6666663f` (`f4d70e4`, CI run `35112569688`) — so
/// the mask declares Sample0|Min|Max and the two edge cases the marker names
/// are present in every oracle capture under the presence-iff-bit rule.
private let nativeDepthResolveModes: UInt64 = (1 << 0) | (1 << 1) | (1 << 2)

/// The stencil resolve capability mask this oracle declares
/// (`research/docs/23` §3.3, v60): bit `i` is the filter whose wire code is
/// `i`. The v59 `--stencil-resolve-selftest` run measured the Apple
/// Paravirtual device executing both filters — the mixed column lands
/// `depth_resolved_sample(min)=01` and `depth_resolved_sample(max)=00`
/// (`2b877b8`, CI run `35120171655`) — so the mask declares
/// Sample0|DepthResolvedSample and the gated case the marker names is present
/// in every oracle capture under the presence-iff-bit rule.
private let nativeStencilResolveModes: UInt64 = (1 << 0) | (1 << 1)

private struct SuiteResult: Encodable {
    let schema_version: UInt64
    let suite: String
    let suite_sha256: String
    let backend: String
    let allocation_observation: String
    let depth_resolve_modes: UInt64
    let stencil_resolve_modes: UInt64
    let device: String
    let platform: String
    let results: [CaseResult]
}

/// The one-device heap check's report (`research/docs/25` §6 Step 7a). It
/// reuses the same writeback/allocation observation shape every other report
/// uses, and carries the device and platform so the CI step can assert the
/// observations came from the probed device rather than a fixture.
private struct HeapSelfTestReport: Encodable {
    let id: String
    let completion: String
    let writebacks: [Writeback]
    let allocations: [AllocationResult]
    let device: String
    let platform: String
}

private struct DeviceProbe: Encodable {
    let schema_version: UInt64 = 1
    let kind = "metal-device-probe"
    let platform: String
    let device: String?
    let eligible: Bool
    let reason: String
    let supports_apple4: Bool
    let has_unified_memory: Bool

    private enum CodingKeys: String, CodingKey {
        case schema_version, kind, platform, device, eligible, reason
        case supports_apple4, has_unified_memory
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(schema_version, forKey: .schema_version)
        try container.encode(kind, forKey: .kind)
        try container.encode(platform, forKey: .platform)
        // An absent device is explicitly null, not an omitted schema field.
        try container.encode(device, forKey: .device)
        try container.encode(eligible, forKey: .eligible)
        try container.encode(reason, forKey: .reason)
        try container.encode(supports_apple4, forKey: .supports_apple4)
        try container.encode(has_unified_memory, forKey: .has_unified_memory)
    }
}

private struct Options {
    let suite: URL?
    let output: URL?
    let validateOnly: Bool
    let probe: Bool
    let renderSelfTest: Bool
    let presentSelfTest: Bool
    let vertexSelfTest: Bool
    let mrtSelfTest: Bool
    let heapSelfTest: Bool
    let depthResolveSelfTest: Bool
    let stencilResolveSelfTest: Bool
}

private let usage = """
Usage: native-metal-oracle --suite PATH [--output PATH]
       native-metal-oracle --suite PATH --validate-suite
       native-metal-oracle --probe
       native-metal-oracle --render-selftest
       native-metal-oracle --present-selftest
       native-metal-oracle --vertex-selftest
       native-metal-oracle --mrt-selftest
       native-metal-oracle --heap-selftest
       native-metal-oracle --depth-resolve-selftest
       native-metal-oracle --stencil-resolve-selftest
       native-metal-oracle --help

Capture the supported suite using native Metal on Apple silicon macOS 11+.
Without --output, the successful JSON report goes to stdout. Existing output
files are never overwritten. Diagnostics go to stderr. --validate-suite checks
the fixture and both shader source hashes without creating a Metal device.
--probe needs no suite and reports default-device eligibility as JSON to stdout.
It cannot be combined with other options. Probe success means the query succeeded;
it does not mean a device is eligible or that any Metal compute work executed.
--render-selftest needs no suite: it captures the reviewed 2x2 offscreen render
fixture, resolved relative to the current working directory, and prints the
observed attachment bytes as JSON. It fails unless all four texels read back as
the reviewed fragment output rather than the clear sentinel, and it cannot be
combined with other options.
--present-selftest needs no suite: it captures the reviewed 2x2 present
equivalent, resolving the same module relative to the current working
directory. The present target is preset with the fefefefe sentinel, the
reviewed fragment draws over it, and the report fails unless all four texels
read back as the fragment output rather than the sentinel. It cannot be
combined with other options.
--vertex-selftest needs no suite: it captures the reviewed indexed 2x2 quad,
resolving the indexed module (shaders/quad_indexed_2x2.metal) relative to the
current working directory. The fixture's float32x2 stream and uint16 index
buffer are bound through an MTLVertexDescriptor and drawn with
drawIndexedPrimitives, and the report fails unless all four texels read back as
the reviewed fragment output rather than the clear sentinel. It cannot be
combined with other options.
--mrt-selftest needs no suite: it captures the reviewed dual-output 2x2 quad,
resolving the dual module (shaders/quad_indexed_2x2_dual.metal) relative to the
current working directory. The same stream and index buffer as --vertex-selftest
are drawn into two colour attachments, and the report fails unless location 0
reads back 4080c0ff and location 1 ff8040c0, never the clear sentinel. It cannot
be combined with other options.
--heap-selftest needs no suite: it allocates two reviewed heap buffers from one
MTLHeap, records their heap offsets, runs the reviewed copy_word kernel across
the pair, and reports the copied bytes. It fails unless both buffers are in the
same heap with non-overlapping ranges and the write buffer reads back the
reviewed word rather than the sentinel. It cannot be combined with other
options.
--depth-resolve-selftest needs no suite: it renders the reviewed depth pair's
v51 edge geometry (a near triangle at z = 0.5 covering NDC x <= 0.25, a far
triangle at z = 0.9 covering everything, one red tint) three times through a
four-sample depth32float raster cleared to 1.0, once per depth resolve filter
(sample0, min, max), and prints each texel's three resolved depth landings one
line at a time. It prints PASS only when the stable columns are as reviewed
(column 0 = 0.5 and column 3 = 0.9 for every filter) and the key column
distinguishes the filters (column 2 = 0.5 for min, 0.9 for max); sample0's
column 2 is recorded but not judged because it depends on the rasterizer's
sample positions. It cannot be combined with other options.
--stencil-resolve-selftest needs no suite: it renders the reviewed depth pair's
v51 edge geometry (a near triangle at z = 0.5 covering NDC x <= 0.25, a far
triangle at z = 0.9 covering everything) three times through a four-sample
depth32float-stencil8 raster cleared to depth 1.0 and stencil 0. The near
triangle passes an equal-0 test and increments-wraps to stencil 1; the far
triangle covers the remaining samples without writing stencil, so every texel
whose samples straddle x = 0.25 carries both a stencil-1 near sample and a
stencil-0 far sample. The three passes print one stencil8 texel per line: the
sample0 stencil filter, the depthResolvedSample filter with a min depth
resolve, and the depthResolvedSample filter with a max depth resolve. It prints
PASS only when the stable columns hold (column 0 = 01 and column 3 = 00 for
every filter) and the key column distinguishes the depth-resolved sample
(column 2 = 01 for min, 00 for max); sample0's column 2 is recorded but not
judged because it depends on the rasterizer's sample positions. It cannot be
combined with other options.
The 20-second completion timeout does not cancel submitted GPU work.
"""

private func parseOptions(_ arguments: [String]) throws -> Options {
    var suite: URL?
    var output: URL?
    var validateOnly = false
    var probe = false
    var renderSelfTest = false
    var presentSelfTest = false
    var vertexSelfTest = false
    var mrtSelfTest = false
    var heapSelfTest = false
    var depthResolveSelfTest = false
    var stencilResolveSelfTest = false
    var index = 0
    while index < arguments.count {
        let argument = arguments[index]
        switch argument {
        case "--suite", "--output":
            try require(index + 1 < arguments.count, "Missing value for \(argument)")
            let value = arguments[index + 1]
            try require(!value.isEmpty && !value.hasPrefix("--"), "Missing path for \(argument)")
            let url = URL(fileURLWithPath: value).standardizedFileURL
            if argument == "--suite" {
                try require(suite == nil, "Duplicate --suite option")
                suite = url
            } else {
                try require(output == nil, "Duplicate --output option")
                output = url
            }
            index += 2
        case "--validate-suite":
            try require(!validateOnly, "Duplicate --validate-suite option")
            validateOnly = true
            index += 1
        case "--probe":
            try require(!probe, "Duplicate --probe option")
            probe = true
            index += 1
        case "--render-selftest":
            try require(!renderSelfTest, "Duplicate --render-selftest option")
            renderSelfTest = true
            index += 1
        case "--present-selftest":
            try require(!presentSelfTest, "Duplicate --present-selftest option")
            presentSelfTest = true
            index += 1
        case "--vertex-selftest":
            try require(!vertexSelfTest, "Duplicate --vertex-selftest option")
            vertexSelfTest = true
            index += 1
        case "--mrt-selftest":
            try require(!mrtSelfTest, "Duplicate --mrt-selftest option")
            mrtSelfTest = true
            index += 1
        case "--heap-selftest":
            try require(!heapSelfTest, "Duplicate --heap-selftest option")
            heapSelfTest = true
            index += 1
        case "--depth-resolve-selftest":
            try require(!depthResolveSelfTest, "Duplicate --depth-resolve-selftest option")
            depthResolveSelfTest = true
            index += 1
        case "--stencil-resolve-selftest":
            try require(!stencilResolveSelfTest, "Duplicate --stencil-resolve-selftest option")
            stencilResolveSelfTest = true
            index += 1
        default:
            throw OracleError("Unknown argument: \(argument)\n\(usage)")
        }
    }
    if probe {
        try require(suite == nil && output == nil && !validateOnly && !renderSelfTest && !presentSelfTest && !vertexSelfTest && !mrtSelfTest && !heapSelfTest && !depthResolveSelfTest && !stencilResolveSelfTest,
                    "--probe cannot be combined with --suite, --output, --validate-suite, --render-selftest, --present-selftest, --vertex-selftest, --mrt-selftest, --heap-selftest, --depth-resolve-selftest, or --stencil-resolve-selftest")
        return Options(suite: nil, output: nil, validateOnly: false, probe: true,
                       renderSelfTest: false, presentSelfTest: false, vertexSelfTest: false,
                       mrtSelfTest: false, heapSelfTest: false, depthResolveSelfTest: false,
                       stencilResolveSelfTest: false)
    }
    if renderSelfTest {
        try require(suite == nil && output == nil && !validateOnly && !presentSelfTest && !vertexSelfTest && !mrtSelfTest && !heapSelfTest && !depthResolveSelfTest && !stencilResolveSelfTest,
                    "--render-selftest cannot be combined with --suite, --output, --validate-suite, --present-selftest, --vertex-selftest, --mrt-selftest, --heap-selftest, --depth-resolve-selftest, or --stencil-resolve-selftest")
        return Options(suite: nil, output: nil, validateOnly: false, probe: false,
                       renderSelfTest: true, presentSelfTest: false, vertexSelfTest: false,
                       mrtSelfTest: false, heapSelfTest: false, depthResolveSelfTest: false,
                       stencilResolveSelfTest: false)
    }
    if presentSelfTest {
        try require(suite == nil && output == nil && !validateOnly && !vertexSelfTest && !mrtSelfTest && !heapSelfTest && !depthResolveSelfTest && !stencilResolveSelfTest,
                    "--present-selftest cannot be combined with --suite, --output, --validate-suite, --vertex-selftest, --mrt-selftest, --heap-selftest, --depth-resolve-selftest, or --stencil-resolve-selftest")
        return Options(suite: nil, output: nil, validateOnly: false, probe: false,
                       renderSelfTest: false, presentSelfTest: true, vertexSelfTest: false,
                       mrtSelfTest: false, heapSelfTest: false, depthResolveSelfTest: false,
                       stencilResolveSelfTest: false)
    }
    if vertexSelfTest {
        try require(suite == nil && output == nil && !validateOnly && !mrtSelfTest && !heapSelfTest && !depthResolveSelfTest && !stencilResolveSelfTest,
                    "--vertex-selftest cannot be combined with --suite, --output, --validate-suite, --mrt-selftest, --heap-selftest, --depth-resolve-selftest, or --stencil-resolve-selftest")
        return Options(suite: nil, output: nil, validateOnly: false, probe: false,
                       renderSelfTest: false, presentSelfTest: false, vertexSelfTest: true,
                       mrtSelfTest: false, heapSelfTest: false, depthResolveSelfTest: false,
                       stencilResolveSelfTest: false)
    }
    if mrtSelfTest {
        try require(suite == nil && output == nil && !validateOnly && !heapSelfTest && !depthResolveSelfTest && !stencilResolveSelfTest,
                    "--mrt-selftest cannot be combined with --suite, --output, --validate-suite, --heap-selftest, --depth-resolve-selftest, or --stencil-resolve-selftest")
        return Options(suite: nil, output: nil, validateOnly: false, probe: false,
                       renderSelfTest: false, presentSelfTest: false, vertexSelfTest: false,
                       mrtSelfTest: true, heapSelfTest: false, depthResolveSelfTest: false,
                       stencilResolveSelfTest: false)
    }
    if heapSelfTest {
        try require(suite == nil && output == nil && !validateOnly && !depthResolveSelfTest && !stencilResolveSelfTest,
                    "--heap-selftest cannot be combined with --suite, --output, --validate-suite, --depth-resolve-selftest, or --stencil-resolve-selftest")
        return Options(suite: nil, output: nil, validateOnly: false, probe: false,
                       renderSelfTest: false, presentSelfTest: false, vertexSelfTest: false,
                       mrtSelfTest: false, heapSelfTest: true, depthResolveSelfTest: false,
                       stencilResolveSelfTest: false)
    }
    if depthResolveSelfTest {
        try require(suite == nil && output == nil && !validateOnly && !stencilResolveSelfTest,
                    "--depth-resolve-selftest cannot be combined with --suite, --output, --validate-suite, or --stencil-resolve-selftest")
        return Options(suite: nil, output: nil, validateOnly: false, probe: false,
                       renderSelfTest: false, presentSelfTest: false, vertexSelfTest: false,
                       mrtSelfTest: false, heapSelfTest: false, depthResolveSelfTest: true,
                       stencilResolveSelfTest: false)
    }
    if stencilResolveSelfTest {
        try require(suite == nil && output == nil && !validateOnly,
                    "--stencil-resolve-selftest cannot be combined with --suite, --output, or --validate-suite")
        return Options(suite: nil, output: nil, validateOnly: false, probe: false,
                       renderSelfTest: false, presentSelfTest: false, vertexSelfTest: false,
                       mrtSelfTest: false, heapSelfTest: false, depthResolveSelfTest: false,
                       stencilResolveSelfTest: true)
    }
    try require(suite != nil, "--suite is required\n\(usage)")
    try require(!validateOnly || output == nil, "--output cannot be used with --validate-suite")
    if let outputURL = output {
        try require(!FileManager.default.fileExists(atPath: outputURL.path),
                    "Output already exists: \(outputURL.path)")
    }
    return Options(suite: suite, output: output, validateOnly: validateOnly, probe: false,
                   renderSelfTest: false, presentSelfTest: false, vertexSelfTest: false,
                   mrtSelfTest: false, heapSelfTest: false, depthResolveSelfTest: false,
                   stencilResolveSelfTest: false)
}

private func readBoundedFile(_ url: URL) throws -> Data {
    let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
    try require(attributes[.type] as? FileAttributeType == .typeRegular,
                "Expected a regular file: \(url.path)")
    guard let size = attributes[.size] as? NSNumber else {
        throw OracleError("Cannot determine file size: \(url.path)")
    }
    try require(size.uint64Value <= UInt64(maximumFileBytes), "File exceeds 1 MiB: \(url.path)")
    // Recheck the actual read in case the file changed after the size check.
    let bytes = try Data(contentsOf: url)
    try require(bytes.count <= maximumFileBytes, "File exceeds 1 MiB: \(url.path)")
    return bytes
}

private func hex(_ bytes: Data) -> String {
    let alphabet = Array("0123456789abcdef".utf8)
    var result = [UInt8]()
    result.reserveCapacity(bytes.count * 2)
    for byte in bytes {
        result.append(alphabet[Int(byte >> 4)])
        result.append(alphabet[Int(byte & 15)])
    }
    return String(decoding: result, as: UTF8.self)
}

private func decodeHex(_ value: String, context: String) throws -> Data {
    let characters = Array(value.utf8)
    try require(characters.count <= maximumFileBytes * 2 && characters.count % 2 == 0,
                "Invalid hex byte length in \(context)")
    func nibble(_ character: UInt8) throws -> UInt8 {
        switch character {
        case 48...57: return character - 48
        case 97...102: return character - 97 + 10
        default: throw OracleError("Expected lowercase hexadecimal in \(context)")
        }
    }
    var result = Data()
    result.reserveCapacity(characters.count / 2)
    for index in stride(from: 0, to: characters.count, by: 2) {
        let high = try nibble(characters[index])
        let low = try nibble(characters[index + 1])
        result.append((high << 4) | low)
    }
    return result
}

@available(macOS 11.0, *)
private func sha256(_ bytes: Data) -> String {
    hex(Data(SHA256.hash(data: bytes)))
}

@available(macOS 11.0, *)
private func validateSource(_ definition: SourceDefinition, root: URL,
                            path: String, digest: String) throws -> Data {
    // The source identities are part of the manual footprint proof. Merely
    // updating a fixture hash must not admit an arbitrary shader for execution.
    try require(definition.path == path && definition.sha256 == digest,
                "Unreviewed shader identity: \(definition.path)")
    let bytes = try readBoundedFile(root.appendingPathComponent(path).standardizedFileURL)
    try require(sha256(bytes) == digest, "Shader SHA-256 mismatch: \(path)")
    return bytes
}

private func validateShape(_ definition: CaseDefinition, suite: String,
                           declaringShapeIDs: Set<String>) throws -> [DispatchDefinition] {
    try require(definition.grid.count == 3 && definition.local.count == 3,
                "\(definition.id): grid and local need three dimensions")
    try require(definition.grid.allSatisfy { $0 > 0 && $0 <= 1024 }
                && definition.local.allSatisfy { $0 > 0 && $0 <= 1024 },
                "\(definition.id): dimensions must be in 1...1024")
    try require(definition.local.reduce(UInt64(1), *) <= 1024,
                "\(definition.id): excessive threads per threadgroup")
    let dispatches: [DispatchDefinition]
    if suite == "compute-buffer-v3" || suite == "compute-buffer-v4" || suite == "compute-buffer-v5" || suite == "compute-buffer-v6" || suite == "compute-buffer-v7" || suite == "compute-buffer-v8" || suite == "compute-buffer-v9" {
        let expectedCount: Int
        switch definition.id {
        case "transform_twice", "transform_pingpong_two", "copy_pingpong", "pipeline_chain_two", "layout_chain_two", "subset_chain_two": expectedCount = 2
        case "transform_three_times", "transform_pingpong_three", "pipeline_chain_three", "layout_chain_three": expectedCount = 3
        case "subset_chain_four": expectedCount = 4
        case "transform_eight_times", "transform_pingpong_eight", "pipeline_chain_eight", "layout_chain_eight", "subset_chain_eight": expectedCount = 8
        default: throw OracleError("Unsupported serial case: \(definition.id)")
        }
        guard let sequence = definition.dispatches else {
            throw OracleError("\(definition.id): serial dispatches are required")
        }
        try require(sequence.count == expectedCount && sequence.count <= maximumPassCount,
                    "\(definition.id): unsupported serial pass count")
        let localSizes: [[UInt64]] = [[4, 2, 2], [8, 4, 4], [1, 1, 1]]
        let expected = try (0..<expectedCount).map { index -> DispatchDefinition in
            var mapping: [UInt64]? = nil
            var program: Int? = (suite == "compute-buffer-v5" || suite == "compute-buffer-v6") ? index % 2 : nil
            if suite == "compute-buffer-v7" || suite == "compute-buffer-v8" || suite == "compute-buffer-v9" {
                let views = definition.buffers.map { $0.view }
                try require(views.count == (expectedCount == 2 ? 4 : 5), "Missing subset-chain resources")
                switch index % 4 {
                case 0: mapping = [views[0], views[1], views[2]]; program = 0
                case 1: mapping = [views[2], views[3]]; program = 1
                case 2: mapping = [views[1], views[3], views[0]]; program = 2
                default: mapping = [views[0], views[4]]; program = 1
                }
            } else if suite == "compute-buffer-v6" {
                let views = definition.buffers.map { $0.view }
                try require(views.count == 3, "Missing layout-chain resources")
                mapping = index % 2 == 0 ? views : [views[1], views[2], views[0]]
            } else if suite == "compute-buffer-v4" || suite == "compute-buffer-v5" {
                var views = definition.buffers.map { $0.view }
                let last = definition.id == "copy_pingpong" ? 1 : 2
                try require(views.count > last, "Missing pingpong resources")
                if index % 2 == 1 { views.swapAt(0, last) }
                mapping = views
            }
            let grid: [UInt64] = definition.id == "copy_pingpong" ? [1, 1, 1] : [5, 3, 2]
            let local: [UInt64] = definition.id == "copy_pingpong" ? [1, 1, 1] : localSizes[index % localSizes.count]
            return DispatchDefinition(grid: grid, local: local, bindings: mapping, program: program)
        }
        try require(sequence == expected, "\(definition.id): unsupported serial dispatch sequence")
        try require(sequence[0].grid == definition.grid && sequence[0].local == definition.local,
                    "\(definition.id): grid/local must match the first serial dispatch")
        dispatches = sequence
    } else {
        try require(definition.dispatches == nil,
                    "\(definition.id): serial dispatches require compute-buffer-v3")
        dispatches = [DispatchDefinition(grid: definition.grid, local: definition.local, bindings: nil, program: nil)]
    }
    switch definition.id {
    case "render_declaring_quad_extent":
        // v27's declaring case: the reviewed copy_word kernel over a 4x4
        // attachment view (64 bytes) and a 4-byte output view.
        try require(definition.entry == "copy_word"
                    && definition.grid == [1, 1, 1] && definition.local == [1, 1, 1],
                    "\(definition.id): unsupported entry or dispatch shape")
        try require(definition.buffers.count == 2, "\(definition.id): expected two buffers")
        try require(definition.buffers.contains { $0.binding == 0 && $0.access == "read" && $0.length == 64 }
                    && definition.buffers.contains { $0.binding == 1 && $0.access == "write" && $0.length == 4 },
                    "\(definition.id): expected a 64-byte read buffer at 0 and a write buffer at 1")
    case "render_declaring_depth_store":
        // v43's declaring case: the reviewed copy_word_with_witness kernel
        // reads one word from each of the two declaring views — the colour
        // attachment's 64 bytes and the depth attachment's 64 bytes, which
        // travel through the same writeback channel (`research/docs/23` §3.3,
        // v43) — and writes both into its 4-byte output view.
        try require(definition.entry == "copy_word_with_witness"
                    && definition.grid == [1, 1, 1] && definition.local == [1, 1, 1],
                    "\(definition.id): unsupported entry or dispatch shape")
        try require(definition.buffers.count == 3, "\(definition.id): expected three buffers")
        try require(definition.buffers.contains { $0.binding == 0 && $0.access == "read" && $0.length == 64 }
                    && definition.buffers.contains { $0.binding == 1 && $0.access == "write" && $0.length == 4 }
                    && definition.buffers.contains { $0.binding == 2 && $0.access == "read" && $0.length == 64 },
                    "\(definition.id): expected a 64-byte read buffer at 0, a write buffer at 1 "
                    + "and a 64-byte read buffer at 2")
    case "render_declaring_depth_resolve":
        // v57d's declaring case: the depth-store sibling's own shape — the
        // reviewed `copy_word_with_witness` kernel over the colour view (64
        // bytes), the 4-byte output view and the second depth landing the
        // Min/Max pairs resolve into (64 bytes), which travels through the same
        // writeback channel (`research/docs/23` §3.3, v57d).
        try require(definition.entry == "copy_word_with_witness"
                    && definition.grid == [1, 1, 1] && definition.local == [1, 1, 1],
                    "\(definition.id): unsupported entry or dispatch shape")
        try require(definition.buffers.count == 3, "\(definition.id): expected three buffers")
        try require(definition.buffers.contains { $0.binding == 0 && $0.access == "read" && $0.length == 64 }
                    && definition.buffers.contains { $0.binding == 1 && $0.access == "write" && $0.length == 4 }
                    && definition.buffers.contains { $0.binding == 2 && $0.access == "read" && $0.length == 64 },
                    "\(definition.id): expected a 64-byte read buffer at 0, a write buffer at 1 "
                    + "and a 64-byte read buffer at 2")
    case "render_declaring_stencil_store":
        // v49's declaring case: the same reviewed copy_word_with_witness
        // kernel over the same colour view and output view, but the third
        // binding is the stencil landing's own view — one byte per `stencil8`
        // texel, so the 4x4 surface's 16 bytes where the depth sibling's
        // `depth32float` view carries 64 (`research/docs/23` §3.3, v49).
        try require(definition.entry == "copy_word_with_witness"
                    && definition.grid == [1, 1, 1] && definition.local == [1, 1, 1],
                    "\(definition.id): unsupported entry or dispatch shape")
        try require(definition.buffers.count == 3, "\(definition.id): expected three buffers")
        try require(definition.buffers.contains { $0.binding == 0 && $0.access == "read" && $0.length == 64 }
                    && definition.buffers.contains { $0.binding == 1 && $0.access == "write" && $0.length == 4 }
                    && definition.buffers.contains { $0.binding == 2 && $0.access == "read" && $0.length == 16 },
                    "\(definition.id): expected a 64-byte read buffer at 0, a write buffer at 1 "
                    + "and a 16-byte read buffer at 2")
    case "render_declaring_stencil_resolve":
        // v60's declaring case: the four-binding sibling of the reviewed
        // `copy_word_with_witness` kernel, whose third and fourth bindings are
        // the combined shape's two landings — the depth view (64 bytes) and the
        // stencil view (16 bytes), both read (`research/docs/23` §3.3, v60).
        try require(definition.entry == "copy_word_with_witnesses"
                    && definition.grid == [1, 1, 1] && definition.local == [1, 1, 1],
                    "\(definition.id): unsupported entry or dispatch shape")
        try require(definition.buffers.count == 4, "\(definition.id): expected four buffers")
        try require(definition.buffers.contains { $0.binding == 0 && $0.access == "read" && $0.length == 64 }
                    && definition.buffers.contains { $0.binding == 1 && $0.access == "write" && $0.length == 4 }
                    && definition.buffers.contains { $0.binding == 2 && $0.access == "read" && $0.length == 64 }
                    && definition.buffers.contains { $0.binding == 3 && $0.access == "read" && $0.length == 16 },
                    "\(definition.id): expected a 64-byte read buffer at 0, a write buffer at 1, "
                    + "a 64-byte read buffer at 2 and a 16-byte read buffer at 3")
    case "copy_word", "copy_seed_a", "copy_seed_b", "copy_pingpong",
         "alias_disjoint_pair", "alias_disjoint_pair_reversed":
        try require(definition.entry == "copy_word"
                    && definition.grid == [1, 1, 1] && definition.local == [1, 1, 1],
                    "copy_word: unsupported entry or dispatch shape")
        try require(definition.buffers.count == 2, "copy_word: expected two buffers")
        try require(definition.buffers.contains { $0.binding == 0 && $0.access == "read" && $0.length == 4 }
                    && definition.buffers.contains { $0.binding == 1 && $0.access == "write" && $0.length == 4 },
                    "copy_word: expected a 4-byte read buffer at 0 and write buffer at 1")
    case "render_declaring_three_attachments", "render_declaring_four_attachments":
        // v24's declaring case: the reviewed mrt_declare4 kernel reads one word
        // from each of the four attachment views and writes their xor into its
        // own output view, so one submission proves it read every view the
        // render pass then writes.
        try require(definition.entry == "mrt_declare4"
                    && definition.grid == [1, 1, 1] && definition.local == [1, 1, 1],
                    "\(definition.id): unsupported entry or dispatch shape")
        try require(definition.buffers.count == 5,
                    "\(definition.id): expected five buffers")
        // The three-attachment sibling keeps only three attachment views; its
        // fourth read is a 4-byte scratch view, because a view the render pass
        // does not attach has to keep its guard bytes.
        let attachmentReads = definition.id == "render_declaring_three_attachments" ? 3 : 4
        try require((0..<attachmentReads).allSatisfy { binding in
            definition.buffers.contains {
                $0.binding == binding && $0.access == "read" && $0.length == 16
            }
        }, "\(definition.id): expected 16-byte attachment read buffers")
        if attachmentReads == 3 {
            try require(definition.buffers.contains {
                $0.binding == 3 && $0.access == "read" && $0.length == 4
            }, "\(definition.id): expected a 4-byte scratch read buffer at 3")
        }
        try require(definition.buffers.contains {
            $0.binding == 4 && $0.access == "write" && $0.length == 4
        }, "\(definition.id): expected a 4-byte write buffer at 4")
    case "render_declaring_two_attachments", "render_declaring_store_and_discard":
        // v18's declaring case and v19's store/discard sibling: the reviewed
        // mrt_declare kernel reads both attachment views and writes their xor
        // into its own output view, so one submission proves it read the two
        // views the render pass names (stored or discarded).
        try require(definition.entry == "mrt_declare"
                    && definition.grid == [1, 1, 1] && definition.local == [1, 1, 1],
                    "\(definition.id): unsupported entry or dispatch shape")
        try require(definition.buffers.count == 3,
                    "\(definition.id): expected three buffers")
        try require(definition.buffers.contains { $0.binding == 0 && $0.access == "read" && $0.length == 16 }
                    && definition.buffers.contains { $0.binding == 1 && $0.access == "read" && $0.length == 16 }
                    && definition.buffers.contains { $0.binding == 2 && $0.access == "write" && $0.length == 4 },
                    "\(definition.id): expected two 16-byte read buffers and a 4-byte write buffer")
    case let id where declaringShapeIDs.contains(id):
        // v13's declaring case: the same reviewed copy kernel, but its read
        // view covers the 16 attachment bytes the render case stores into, so
        // the attachment is reported through the existing writeback channel
        // (`conformance/RENDER-CAPTURE.md` §3) rather than a new one.
        try require(definition.entry == "copy_word"
                    && definition.grid == [1, 1, 1] && definition.local == [1, 1, 1],
                    "\(definition.id): unsupported entry or dispatch shape")
        try require(definition.buffers.count == 2,
                    "\(definition.id): expected two buffers")
        try require(definition.buffers.contains { $0.binding == 0 && $0.access == "read" && $0.length == 16 }
                    && definition.buffers.contains { $0.binding == 1 && $0.access == "write" && $0.length == 4 },
                    "\(definition.id): expected a 16-byte read buffer at 0 and a 4-byte write buffer at 1")
    case "indexed_boundary", "indexed_tail", "indexed_full", "indexed_small_grid", "indexed_unit":
        let expectedLocal: [UInt64]
        switch definition.id {
        case "indexed_boundary", "indexed_tail": expectedLocal = [8, 2, 1]
        case "indexed_full": expectedLocal = [5, 3, 1]
        case "indexed_small_grid": expectedLocal = [16, 4, 1]
        case "indexed_unit": expectedLocal = [1, 1, 1]
        default: throw OracleError("Unsupported indexed case: \(definition.id)")
        }
        try require(definition.entry == "kernel_dispatch_threads_boundary_barrier"
                    && definition.grid == [10, 3, 1] && definition.local == expectedLocal,
                    "indexed_boundary: unsupported entry or dispatch shape")
        try require(definition.buffers.count == 1, "indexed_boundary: expected one buffer")
        let buffer = definition.buffers[0]
        try require(buffer.binding == 0 && buffer.access == "write" && buffer.length == 120,
                    "indexed_boundary: expected a 120-byte write buffer at 0")
    case "transform_tail", "transform_small_grid", "transform_twice", "transform_three_times", "transform_eight_times",
         "transform_pingpong_two", "transform_pingpong_three", "transform_pingpong_eight",
         "pipeline_chain_two", "pipeline_chain_three", "pipeline_chain_eight",
         "layout_chain_two", "layout_chain_three", "layout_chain_eight":
        let expectedLocal: [UInt64] = definition.id == "transform_small_grid" ? [8, 4, 4] : [4, 2, 2]
        try require(definition.entry == "transform_3d"
                    && definition.grid == [5, 3, 2] && definition.local == expectedLocal,
                    "\(definition.id): unsupported entry or dispatch shape")
        try require(definition.buffers.count == 3, "\(definition.id): expected three buffers")
        try require(definition.buffers.contains { $0.binding == 0 && $0.access == "read_write" && $0.length == 120 }
                    && definition.buffers.contains { $0.binding == 2 && $0.access == "read" && $0.length == 4 }
                    && definition.buffers.contains { $0.binding == 5 && $0.access == "write" && $0.length == 120 },
                    "\(definition.id): expected 120-byte read/write at 0, 4-byte read at 2, and 120-byte write at 5")
    case "subset_chain_two", "subset_chain_four", "subset_chain_eight":
        try require((suite == "compute-buffer-v7" || suite == "compute-buffer-v8" || suite == "compute-buffer-v9") && definition.entry == "transform_3d"
                    && definition.grid == [5, 3, 2] && definition.local == [4, 2, 2],
                    "\(definition.id): unsupported entry or dispatch shape")
        let expectedLabels: [UInt64] = definition.id == "subset_chain_two" ? [0, 2, 5, 8] : [0, 2, 5, 8, 9]
        try require(definition.buffers.map { $0.binding } == expectedLabels,
                    "\(definition.id): unsupported resource pool labels")
        for (index, resource) in definition.buffers.enumerated() {
            let access = index == 0 ? "read_write" : index == 1 ? "read" : "write"
            let length: UInt64 = index == 1 ? 4 : 120
            try require(resource.access == access && resource.length == length,
                        "\(definition.id): unsupported resource pool shape")
        }
    case "sampled_texture_first_texel":
        try require(suite == "compute-buffer-v11" && definition.entry == "read_texture_2d"
                    && definition.grid == [1, 1, 1] && definition.local == [1, 1, 1],
                    "\(definition.id): unsupported entry or dispatch shape")
        try require(definition.buffers.count == 1
                    && definition.buffers[0].binding == 0
                    && definition.buffers[0].access == "write"
                    && definition.buffers[0].length == 64,
                    "\(definition.id): expected one 64-byte write-only output buffer")
        try require(definition.textures?.count == 1
                    && definition.textures?[0].binding == 0
                    && definition.textures?[0].width == 4
                    && definition.textures?[0].height == 4
                    && definition.textures?[0].access == "sampled",
                    "\(definition.id): expected one 4x4 sampled texture")
    case "texture_cell_local_4x4", "texture_cell_local_1x1":
        let expectedLocal: [UInt64] = definition.id == "texture_cell_local_4x4" ? [4, 4, 1] : [1, 1, 1]
        try require(suite == "compute-buffer-v12" && definition.entry == "read_texture_2d_cell"
                    && definition.grid == [4, 4, 1] && definition.local == expectedLocal,
                    "\(definition.id): unsupported entry or dispatch shape")
        try require(definition.buffers.count == 1
                    && definition.buffers[0].binding == 0
                    && definition.buffers[0].access == "write"
                    && definition.buffers[0].length == 64,
                    "\(definition.id): expected one 64-byte write-only output buffer")
        try require(definition.textures?.count == 1
                    && definition.textures?[0].binding == 0
                    && definition.textures?[0].width == 4
                    && definition.textures?[0].height == 4
                    && definition.textures?[0].access == "sampled",
                    "\(definition.id): expected one 4x4 sampled texture")
    default:
        throw OracleError("Unsupported case: \(definition.id)")
    }
    return dispatches
}

// Dispatch indices per command buffer. Legacy fixtures record one command
// buffer; the v9 suite splits the same reviewed sequence across several that
// commit and complete in order.
private func validateCommandBuffers(_ definition: CaseDefinition, suite: String,
                                    dispatches: [DispatchDefinition]) throws -> [[Int]] {
    guard let groups = definition.command_buffers else {
        try require(suite != "compute-buffer-v9",
                    "\(definition.id): v9 fixture requires command buffer groups")
        return [Array(0..<dispatches.count)]
    }
    try require(suite == "compute-buffer-v9",
                "\(definition.id): command buffer groups require compute-buffer-v9")
    try require(groups.count >= 2 && groups.count <= 4,
                "\(definition.id): v9 fixture needs two to four command buffers")
    var expected = 0
    for group in groups {
        try require(!group.isEmpty, "\(definition.id): command buffer group cannot be empty")
        for index in group {
            try require(index == expected,
                        "\(definition.id): command buffer groups must partition the dispatch order")
            expected += 1
        }
    }
    try require(expected == dispatches.count,
                "\(definition.id): command buffer groups must partition the dispatch order")
    return groups
}

// Legacy fixtures use their original interface; v6/v7 choose an independently
// reviewed interface for each selected program. Resource pool order is not a
// shader interface when programs use different slots and access modes, or
// bind only a subset of the declared resources.
private func bufferSlots(_ definition: CaseDefinition, program: Int) -> [BufferSlotDefinition] {
    definition.programs?[program].buffer_slots ?? definition.buffers.map {
        BufferSlotDefinition(binding: $0.binding, access: $0.access, length: $0.length)
    }
}

// Called only after program-table, dispatch-shape, and resource-mapping checks.
private func writableViews(_ definition: CaseDefinition) -> Set<UInt64> {
    let sequence = definition.dispatches ?? [DispatchDefinition(grid: definition.grid, local: definition.local, bindings: nil, program: nil)]
    var result = Set<UInt64>()
    for dispatch in sequence {
        for (index, slot) in bufferSlots(definition, program: dispatch.program ?? 0).enumerated() where slot.access != "read" {
            result.insert(dispatch.bindings?[index] ?? definition.buffers[index].view)
        }
    }
    return result
}

/// The v11 texture section: one 4x4 R32Uint sampled texture per case, whose
/// tightly packed initial bytes become an MTLTexture before execution.
private func validateTextures(_ definition: CaseDefinition) throws -> [ValidatedTexture] {
    let textures = definition.textures ?? []
    var views = Set<UInt64>()
    var bindings = Set<Int>()
    var valid = [ValidatedTexture]()
    for texture in textures {
        try require(texture.format == "r32_uint", "\(definition.id): unsupported texture format")
        try require(texture.access == "sampled", "\(definition.id): unsupported texture access")
        try require(texture.width > 0 && texture.height > 0,
                    "\(definition.id): texture dimensions must be nonzero")
        try require(bindings.insert(texture.binding).inserted && views.insert(texture.view).inserted,
                    "\(definition.id): duplicate texture view or binding")
        let backing = try decodeHex(texture.initial_hex,
                                    context: "\(definition.id) texture \(texture.view)")
        try require(backing.count == texture.width * texture.height * 4,
                    "\(definition.id): texture bytes do not match the declared extent")
        valid.append(ValidatedTexture(definition: texture, backing: backing))
    }
    return valid
}

private func validateBuffers(_ definition: CaseDefinition, guardByte: UInt8,
                             declaringShapeIDs: Set<String>) throws -> [ValidatedBuffer] {
    var bindings = Set<UInt64>()
    var views = Set<UInt64>()
    var buffers = [ValidatedBuffer]()
    // One image per allocation. A v10 fixture binds two disjoint views of the
    // same allocation, and every view of it must be bound against that single
    // image so the reported allocation can be compared as one extent. Guard
    // bytes outside every view stay in the image, which keeps an offset or
    // extent mistake observable.
    var images = [UInt64: Data]()
    var ranges = [UInt64: [(UInt64, UInt64)]]()
    try require(definition.buffers.map { $0.binding } == definition.buffers.map { $0.binding }.sorted(),
                "Bindings must be in canonical order")
    for buffer in definition.buffers {
        let context = "\(definition.id) binding \(buffer.binding)"
        try require(bindings.insert(buffer.binding).inserted, "\(context): duplicate binding")
        try require(views.insert(buffer.view).inserted, "\(context): duplicate view")
        try require(buffer.allocation > 0 && buffer.view > 0, "\(context): zero resource identity")
        try require(buffer.access == "read" || buffer.access == "write" || buffer.access == "read_write",
                    "\(context): unsupported access")
        try require(buffer.length > 0 && buffer.allocation_size <= maximumAllocationBytes,
                    "\(context): allocation must be nonempty and at most 1 MiB")
        try require(buffer.offset <= buffer.allocation_size
                    && buffer.length <= buffer.allocation_size - buffer.offset,
                    "\(context): view extends beyond allocation")
        try require(buffer.offset % 4 == 0, "\(context): uint binding offset needs 4-byte alignment")
        // Each owned view must have a canary prefix and suffix to make an
        // offset/extent mismatch observable. Bounds above make addition safe.
        //
        // The declaring pass is the one exception: its read view starts at the
        // attachment allocation's first byte (`offset == 0`) and covers
        // exactly the bytes the render case's own pass stores into. For
        // v13–v43 that view *is* the whole attachment allocation
        // (`allocation_size == length`); v49's stencil landing is the 16-byte
        // prefix of a 64-byte allocation instead — the 4x4 `stencil8` surface
        // where the depth sibling's `depth32float` surface fills its own
        // allocation (`research/docs/23` §3.3, v49). There is no neighbouring
        // byte on the left to guard with in either shape, the trailing bytes of
        // a prefix-shaped allocation stay at the guard byte, and the render
        // path's own sentinel-versus-fragment check is what keeps an extent
        // mistake observable there (`conformance/RENDER-CAPTURE.md` §3). The
        // exception is written down here rather than loosening the rule for
        // every case.
        let declaringLandingView =
            declaringShapeIDs.contains(definition.id)
            && buffer.access == "read" && buffer.offset == 0
        let end = buffer.offset + buffer.length
        try require(declaringLandingView
                    || (buffer.offset >= 4 && buffer.allocation_size - end >= 4),
                    "\(context): expected at least four guard bytes before and after the view")
        let initial = try decodeHex(buffer.initial_hex, context: context)
        try require(UInt64(initial.count) == buffer.length, "\(context): initial data length mismatch")
        // Several buffers may name one allocation while their byte ranges stay
        // disjoint. Overlapping ranges would make the observed image depend on
        // write order, so they are refused here exactly as provider admission
        // refuses them.
        let existing = ranges[buffer.allocation] ?? []
        try require(!existing.contains { buffer.offset < $0.1 && $0.0 < end },
                    "\(context): overlapping views of allocation \(buffer.allocation)")
        ranges[buffer.allocation] = existing + [(buffer.offset, end)]
        var image = images[buffer.allocation]
            ?? Data(repeating: guardByte, count: Int(buffer.allocation_size))
        try require(image.count == Int(buffer.allocation_size),
                    "\(context): inconsistent allocation size")
        image.replaceSubrange(Int(buffer.offset)..<Int(end), with: initial)
        images[buffer.allocation] = image
    }
    // Every view shares its allocation's final image, so a view never observes
    // a partial initialization of a sibling view.
    for buffer in definition.buffers {
        guard let image = images[buffer.allocation] else {
            throw OracleError("\(definition.id): missing allocation image")
        }
        buffers.append(ValidatedBuffer(definition: buffer, backing: image))
    }

    let written = writableViews(definition)
    let writable = definition.buffers.filter { written.contains($0.view) }
    try require(definition.expected_writebacks.count == writable.count,
                "\(definition.id): expected writeback count mismatch")
    var expectedViews = Set<UInt64>()
    for expected in definition.expected_writebacks {
        try require(expectedViews.insert(expected.view).inserted,
                    "\(definition.id): duplicate expected writeback")
        guard let buffer = writable.first(where: { $0.view == expected.view }) else {
            throw OracleError("\(definition.id): expected writeback does not name a writable view")
        }
        try require(buffer.allocation == expected.allocation && buffer.offset == expected.offset,
                    "\(definition.id): expected writeback metadata mismatch")
        let bytes = try decodeHex(expected.bytes_hex, context: "\(definition.id) expected writeback")
        try require(UInt64(bytes.count) == buffer.length,
                    "\(definition.id): expected writeback length mismatch")
    }
    return buffers
}

@available(macOS 11.0, *)
private func loadSuite(_ url: URL) throws -> ValidatedSuite {
    let raw = try readBoundedFile(url)
    let suite = try JSONDecoder().decode(SuiteDefinition.self, from: raw)
    try require(suite.schema_version == 1, "Only schema version 1 is supported")
    let expectedIDs: Set<String>
    switch suite.suite {
    case "compute-buffer-v1":
        expectedIDs = ["copy_word", "indexed_boundary"]
    case "compute-buffer-v2":
        expectedIDs = ["copy_seed_a", "copy_seed_b", "indexed_tail", "indexed_full",
                       "indexed_small_grid", "indexed_unit", "transform_tail", "transform_small_grid"]
    case "compute-buffer-v3":
        expectedIDs = ["transform_twice", "transform_three_times", "transform_eight_times"]
    case "compute-buffer-v4":
        expectedIDs = ["transform_pingpong_two", "transform_pingpong_three", "transform_pingpong_eight", "copy_pingpong"]
    case "compute-buffer-v5":
        expectedIDs = ["pipeline_chain_two", "pipeline_chain_three", "pipeline_chain_eight"]
    case "compute-buffer-v6":
        expectedIDs = ["layout_chain_two", "layout_chain_three", "layout_chain_eight"]
    case "compute-buffer-v7":
        expectedIDs = ["subset_chain_two", "subset_chain_four", "subset_chain_eight"]
    case "compute-buffer-v8":
        expectedIDs = ["subset_chain_two", "subset_chain_four", "subset_chain_eight"]
    case "compute-buffer-v9":
        expectedIDs = ["subset_chain_two", "subset_chain_four", "subset_chain_eight"]
    case "compute-buffer-v10":
        expectedIDs = ["alias_disjoint_pair", "alias_disjoint_pair_reversed"]
    case "compute-buffer-v11":
        expectedIDs = ["sampled_texture_first_texel"]
    case "compute-buffer-v12":
        expectedIDs = ["texture_cell_local_4x4", "texture_cell_local_1x1"]
    case "compute-buffer-v13":
        expectedIDs = ["render_declaring_copy_word"]
    case "compute-buffer-v14":
        expectedIDs = ["render_declaring_copy_word"]
    case "compute-buffer-v15":
        expectedIDs = ["heap_placement_copy_word", "icb_dispatch_copy_word"]
    case "compute-buffer-v16":
        expectedIDs = ["render_declaring_copy_word"]
    case "compute-buffer-v17":
        expectedIDs = ["render_declaring_copy_word"]
    case "compute-buffer-v18":
        expectedIDs = ["render_declaring_two_attachments"]
    case "compute-buffer-v19":
        expectedIDs = ["render_declaring_store_and_discard"]
    case "compute-buffer-v20":
        expectedIDs = ["render_declaring_copy_word"]
    case "compute-buffer-v21":
        expectedIDs = ["render_declaring_copy_word"]
    case "compute-buffer-v22":
        expectedIDs = ["render_declaring_copy_word"]
    case "compute-buffer-v23":
        expectedIDs = ["render_declaring_four_attachments"]
    case "compute-buffer-v24":
        expectedIDs = ["render_declaring_three_attachments"]
    case "compute-buffer-v25":
        expectedIDs = ["render_declaring_two_attachments"]
    case "compute-buffer-v26":
        expectedIDs = ["render_declaring_quad_extent"]
    case "compute-buffer-v27":
        expectedIDs = ["render_declaring_two_attachments"]
    case "compute-buffer-v28":
        expectedIDs = ["render_declaring_quad_extent", "render_declaring_depth_store",
                       "render_declaring_depth_resolve", "render_declaring_stencil_store",
                       "render_declaring_stencil_resolve"]
    default:
        throw OracleError("Only compute-buffer-v1 through compute-buffer-v28 are supported")
    }
    try require(suite.cases.count == expectedIDs.count && Set(suite.cases.map { $0.id }) == expectedIDs,
                "\(suite.suite): the suite must contain exactly the supported case IDs")
    let root = url.deletingLastPathComponent()
    // The cases a render case names as its declaring pass share the reviewed
    // whole-allocation attachment view, which the guard-byte rule otherwise
    // refuses (`validateShape`'s exception below).
    var declaringCaseIDs = Set((suite.render_cases ?? []).map { $0.declaring_case })
    if suite.suite == "compute-buffer-v15" {
        // v15's heap case is not a render declaring case, but it shares the
        // reviewed whole-allocation read view the render case stores into, so
        // the same shape and the same guard-byte exception apply.
        declaringCaseIDs.formUnion(["heap_placement_copy_word", "icb_dispatch_copy_word"])
    }
    var cases = [ValidatedCase]()
    for definition in suite.cases {
        // Review every program identity and interface before decoding any
        // buffer payload or reading shader files from the supplied manifest.
        let programs = try validatePrograms(definition, suite: suite.suite)
        let dispatches = try validateShape(definition, suite: suite.suite,
                                           declaringShapeIDs: declaringCaseIDs)
        let commandBuffers = try validateCommandBuffers(definition, suite: suite.suite,
                                                        dispatches: dispatches)
        var usedViews = Set<UInt64>()
        for dispatch in dispatches {
            try require(programs.indices.contains(dispatch.program ?? 0), "Unknown selected program")
            let slots = bufferSlots(definition, program: dispatch.program ?? 0)
            let views = dispatch.bindings ?? definition.buffers.map { $0.view }
            try require(views.count == slots.count, "Program binding count mismatch")
            try require(Set(views).count == views.count, "Duplicate resource within one pass")
            for (slot, view) in zip(slots, views) {
                guard let resource = definition.buffers.first(where: { $0.view == view }) else {
                    throw OracleError("Unknown bound resource")
                }
                try require(resource.length == slot.length, "Program binding length mismatch")
                usedViews.insert(view)
            }
        }
        try require(usedViews == Set(definition.buffers.map { $0.view }), "Unused declared resource")
        let buffers = try validateBuffers(definition, guardByte: suite.guard_byte,
                                          declaringShapeIDs: declaringCaseIDs)
        let textures = try validateTextures(definition)
        let loaded = try programs.map { program in
            (definition: program, source: try loadProgram(program, root: root))
        }
        cases.append(ValidatedCase(definition: definition, dispatches: dispatches,
                                   commandBuffers: commandBuffers,
                                   programs: loaded, buffers: buffers,
                                   textures: textures))
    }
    // Render cases are reviewed after the compute cases, and their ids share the
    // same identity space so a report cannot name two cases the same way.
    let renderCases = try loadRenderCases(suite, root: root)
    return ValidatedSuite(name: suite.suite, sha256: sha256(raw), cases: cases,
                          renderCases: renderCases)
}

private struct ReviewedRenderModule {
    let vertex_entry: String
    let fragment_entry: String
    let metal: RenderSourcePin
    /// The vertex layout the module was written for: `nil` for the
    /// `vertex_id` fixture, the reviewed stream list for the indexed one. A
    /// case's declared layout has to equal it, so a fixture cannot widen the
    /// strides or attributes the reviewed module reads.
    let buffers: [RenderVertexBufferLayoutDefinition]?
}

/// The reviewed `vertex_id` render fixture.
///
/// Code-side, like `reviewedProgram`'s table: an updated fixture hash must not
/// be enough to admit a different module for execution. `RenderSourcePin` is a
/// distinct type from `SourceDefinition` on purpose — the suite-coverage check
/// cross-references compute `air`/`metal` pins only, and a render pipeline is
/// one module with two stage entries rather than an air/metal pair.
private func reviewedRenderModule() -> ReviewedRenderModule {
    ReviewedRenderModule(
        vertex_entry: "render_fullscreen_triangle",
        fragment_entry: "render_solid_rgba8",
        metal: RenderSourcePin(path: "shaders/render_offscreen_2x2.metal",
                               sha256: "7430cd19a3497582618226066e95fb6f4ead9071f83b00c53398ccab8ba9d7de"),
        buffers: nil)
}

/// The reviewed indexed render fixture (`research/docs/23` §3.3): the same
/// fragment entry, a vertex stage that reads `[[stage_in]]`, one `float32x2`
/// position attribute at location 0, and an index buffer the draw selects
/// through.
///
/// A second module rather than a second entry pair in one file, because the two
/// shapes need different pipeline state: the `vertex_id` module carries no
/// `MTLVertexDescriptor` at all, while this one is meaningless without the
/// descriptor its layout states. Pinning the layout here is what keeps a
/// fixture from describing a stream the reviewed module does not read.
private func reviewedIndexedModule() -> ReviewedRenderModule {
    ReviewedRenderModule(
        vertex_entry: "render_quad_vertex",
        fragment_entry: "render_solid_rgba8",
        metal: RenderSourcePin(path: "shaders/quad_indexed_2x2.metal",
                               sha256: "aeb662f5d0515ddc4711d821626a72e389506191d11fa03adc9e21ad097379e8"),
        buffers: [RenderVertexBufferLayoutDefinition(
            stride: 8,
            step: "per_vertex",
            attributes: [RenderVertexAttributeDefinition(location: 0, offset: 0,
                                                          format: "float32x2")])])
}

/// The reviewed dual MRT fixture (wave3 R1): the indexed vertex stage plus a
/// fragment stage that writes two colour outputs, compiled against two 2x2
/// `rgba8_unorm` attachments. The second output's texel `ff 80 40 c0` is
/// deliberately the single-output texel with its byte order reversed, so a
/// capture that swapped the two locations reads the wrong bytes.
private func reviewedDualModule() -> ReviewedRenderModule {
    ReviewedRenderModule(
        vertex_entry: "render_quad_vertex",
        fragment_entry: "render_solid_rgba8_dual",
        metal: RenderSourcePin(path: "shaders/quad_indexed_2x2_dual.metal",
                               sha256: "5afc95dd177ba64e3d2e115ab84805fad2b56a7450914a0ab6f8572f26ba7eba"),
        buffers: [RenderVertexBufferLayoutDefinition(
            stride: 8,
            step: "per_vertex",
            attributes: [RenderVertexAttributeDefinition(location: 0, offset: 0,
                                                          format: "float32x2")])])
}

/// The reviewed single-channel float fixture (v22): the indexed vertex stage
/// plus a fragment stage that writes one component, because an `r32float`
/// attachment takes a one-component store. The Vulkan rail carries the same
/// shape as `solid_r32f.frag.spv` and refuses the four-component module for
/// this format; this module is the native half of that review.
private func reviewedR32fModule() -> ReviewedRenderModule {
    ReviewedRenderModule(
        vertex_entry: "render_quad_vertex",
        fragment_entry: "render_solid_r32f",
        metal: RenderSourcePin(path: "shaders/quad_indexed_2x2_r32f.metal",
                               sha256: "2074ee223fe1e472124312b6e1507f383ceddc10a24c3a432e03e61449ed16aa"),
        buffers: [RenderVertexBufferLayoutDefinition(
            stride: 8,
            step: "per_vertex",
            attributes: [RenderVertexAttributeDefinition(location: 0, offset: 0,
                                                          format: "float32x2")])])
}

/// The reviewed four-location fixture (v24): the indexed vertex stage plus a
/// fragment stage that writes every colour location the contract admits. Its
/// four texels are pairwise distinct, so a capture that landed one target twice
/// cannot pass the comparison.
private func reviewedQuadModule() -> ReviewedRenderModule {
    ReviewedRenderModule(
        vertex_entry: "render_quad_vertex",
        fragment_entry: "render_solid_rgba8_quad",
        metal: RenderSourcePin(path: "shaders/quad_indexed_2x2_quad.metal",
                               sha256: "883a3234884c32ccd32ca1dfbfc22cdcefbc6c1a6f5047cbf65bd647a79dfb28"),
        buffers: [RenderVertexBufferLayoutDefinition(
            stride: 8,
            step: "per_vertex",
            attributes: [RenderVertexAttributeDefinition(location: 0, offset: 0,
                                                          format: "float32x2")])])
}

/// The reviewed three-location fixture (v25): the indexed vertex stage plus a
/// fragment stage that writes three colour locations. Three is not the ceiling,
/// so the four-location module cannot stand in for it.
private func reviewedTripleModule() -> ReviewedRenderModule {
    ReviewedRenderModule(
        vertex_entry: "render_quad_vertex",
        fragment_entry: "render_solid_rgba8_triple",
        metal: RenderSourcePin(path: "shaders/quad_indexed_2x2_triple.metal",
                               sha256: "edfdc95fe3336fb457e71379ed64b358a5cb4f007d9ad93c75d7727e62338d26"),
        buffers: [RenderVertexBufferLayoutDefinition(
            stride: 8,
            step: "per_vertex",
            attributes: [RenderVertexAttributeDefinition(location: 0, offset: 0,
                                                          format: "float32x2")])])
}

/// The reviewed instanced fixture (`research/docs/23` §3.3, v31): the reviewed
/// quad's `float32x2` position stream plus a second `float32x4` tint stream
/// that advances once per *instance*, and a vertex stage that reads the tint
/// and shifts each instance's copy of the quad with `instance_id`. The solid
/// modules cannot stand in for it: the tint travels through a varying only
/// this vertex stage produces, and the fragment stage that stores it is the
/// same review surface as the pair.
///
/// The step pair is part of the pin rather than a knob: binding 0 advances per
/// vertex and binding 1 per instance, which is what the two halves of the
/// attachment observe at once.
private func reviewedInstancedModule() -> ReviewedRenderModule {
    ReviewedRenderModule(
        vertex_entry: "render_instanced_quad_vertex",
        fragment_entry: "render_instanced_tint",
        metal: RenderSourcePin(path: "shaders/instanced_quad_2x2.metal",
                               sha256: "5d22083c7a13f42bd20d7ba85aa1ef5d433409caf13a1d792fd89d76fcef5aab"),
        buffers: [
            RenderVertexBufferLayoutDefinition(
                stride: 8,
                step: "per_vertex",
                attributes: [RenderVertexAttributeDefinition(location: 0, offset: 0,
                                                              format: "float32x2")]),
            RenderVertexBufferLayoutDefinition(
                stride: 16,
                step: "per_instance",
                attributes: [RenderVertexAttributeDefinition(location: 1, offset: 0,
                                                              format: "float32x4")]),
        ])
}

/// The reviewed depth fixture (`research/docs/23` §3.3, v36): one stride-32
/// per-vertex stream whose vertices carry a `float32x3` position at offset 0 —
/// the *caller* chooses each triangle's depth — and a `float32x4` tint at
/// offset 16, with a stage pair that forwards the tint to the attachment. The
/// solid modules cannot stand in for it: the position is a caller-held
/// attribute rather than `vertex_id`, two attributes share one vertex, and the
/// fragment stage stores the tint the vertex stage forwarded. The cull pair
/// (`research/docs/23` §3.3, v39) draws the same stream shape with its own
/// encoder state, so this one module serves both reviewed pair fixtures.
private func reviewedDepthModule() -> ReviewedRenderModule {
    ReviewedRenderModule(
        vertex_entry: "render_depth_pair_vertex",
        fragment_entry: "render_depth_pair_tint",
        metal: RenderSourcePin(path: "shaders/depth_pair_4x4.metal",
                               sha256: "726b6fe282e3ad91f5e4df826ed709dafd53d8a8beac06087490e54656720271"),
        buffers: [RenderVertexBufferLayoutDefinition(
            stride: 32,
            step: "per_vertex",
            attributes: [RenderVertexAttributeDefinition(location: 0, offset: 0,
                                                          format: "float32x3"),
                         RenderVertexAttributeDefinition(location: 1, offset: 16,
                                                          format: "float32x4")])])
}

/// The reviewed zero-colour-attachment depth fixture (`research/docs/23` §3.3,
/// v46): the depth pair's own stride-32 two-attribute stream, drawn by a stage
/// pair whose fragment entry is **void**. A pass with no colour attachment at
/// all is well formed exactly this way — the Metal Shading Language
/// Specification states it as "If the fragment function does not generate
/// output, it returns void", and `MTLRenderPipelineDescriptor.fragmentFunction`
/// documents the effect: no writes to the colour render target occur, while
/// depth (and stencil) writes still proceed. So the rasterizer tests and writes
/// depth for every covered fragment, and the stored depth surface is the whole
/// observation.
///
/// The depth pair's module cannot stand in for it: `render_depth_pair_tint`
/// generates a colour output that no attachment of this shape declares, so the
/// reviewed module has to be the one whose fragment stage generates nothing.
/// The stream shape is the pair's own, which is what lets the case keep the
/// same vertex bytes its neighbours read.
private func reviewedDepthOnlyModule() -> ReviewedRenderModule {
    ReviewedRenderModule(
        vertex_entry: "render_depth_only_vertex",
        fragment_entry: "render_depth_only_fragment",
        metal: RenderSourcePin(path: "shaders/depth_only_4x4.metal",
                               sha256: "9e66c6059a5ff04db5e1bfe221ec1b5b6fc3664c0d555196ca8c903310571dcb"),
        buffers: [RenderVertexBufferLayoutDefinition(
            stride: 32,
            step: "per_vertex",
            attributes: [RenderVertexAttributeDefinition(location: 0, offset: 0,
                                                          format: "float32x3"),
                         RenderVertexAttributeDefinition(location: 1, offset: 16,
                                                          format: "float32x4")])])
}

/// The reviewed module a render case's vertex-input and colour-format shapes
/// select, mirroring `crates/metal-api-native/src/render.rs::reviewed_module`:
/// a `vertex_id` single-attachment case draws the triangle module, a
/// single-attachment case whose one stream carries two attributes draws the
/// pair module the depth and cull fixtures share, a single-attachment case
/// whose two-stream layout steps per instance draws the instanced module, a
/// single-attachment case with any other layout the indexed one, and an
/// indexed case with two `rgba8_unorm` attachments the dual one. The
/// zero-colour shape (`§3.3`, v46) carries no attachment at all, and its
/// two-attribute stream selects the void-fragment module the depth pair's own
/// layout names. A shape no module was reviewed for is refused instead of
/// matched approximately.
private func reviewedModule(for definition: RenderCaseDefinition) throws -> ReviewedRenderModule {
    let attachments = try colorAttachments(definition)
    // The 8-bit UNORM modules are layout-agnostic: the same store lands in
    // whichever channel order each attachment declares, so any mix of the two
    // 8-bit formats is served by the module of its attachment count
    // (`research/docs/23` §3.3, v26).
    let unorm8 = { (format: String) in
        format == "rgba8_unorm" || format == "bgra8_unorm"
    }
    // The instanced shape (`research/docs/23` §3.3, v31) is the one two-stream
    // layout: binding 0 advances per vertex and binding 1 once per instance.
    // The reviewed equality check below pins the rest of the layout, so this
    // selection only has to find the shape's own module.
    let instanced = definition.vertex_layout.map { layout -> Bool in
        layout.buffers.count == 2
            && layout.buffers.contains(where: { $0.resolvedStep == "per_instance" })
    } ?? false
    switch (definition.vertex_layout, attachments.count) {
    case (nil, 1):
        return reviewedRenderModule()
    // The reviewed zero-colour shape (`research/docs/23` §3.3, v46) is the
    // depth pair's own single stream drawn with no colour attachment at all.
    // The attachment count tells it apart from the v36 pair, whose layout is
    // otherwise the same; the fragment entry it selects is the void one, which
    // is what makes a pass without a colour target well formed.
    case (let layout?, 0) where layout.buffers.count == 1
        && layout.buffers[0].attributes.count == 2:
        return reviewedDepthOnlyModule()
    // The reviewed depth shape (`research/docs/23` §3.3, v36) is the one
    // single-stream layout carrying two attributes: a `float32x3` position and
    // a `float32x4` tint sharing one stride-32 vertex. The reviewed equality
    // check below pins the rest of the layout, so this selection only has to
    // find the shape's own module; the cull pair (`§3.3`, v39) carries the
    // same layout and so selects the same module.
    case (let layout?, 1) where layout.buffers.count == 1
        && layout.buffers[0].attributes.count == 2:
        return reviewedDepthModule()
    case (_?, 1) where attachments[0].format == "r32float":
        return reviewedR32fModule()
    case (_?, 1) where instanced && unorm8(attachments[0].format):
        return reviewedInstancedModule()
    case (_?, 1) where unorm8(attachments[0].format):
        return reviewedIndexedModule()
    case (_?, 2) where attachments.allSatisfy({ unorm8($0.format) }):
        return reviewedDualModule()
    case (_?, 3) where attachments.allSatisfy({ unorm8($0.format) }):
        return reviewedTripleModule()
    case (_?, 4) where attachments.allSatisfy({ unorm8($0.format) }):
        return reviewedQuadModule()
    default:
        throw OracleError("\(definition.id): no reviewed module carries this "
                          + "vertex-input and colour-format shape")
    }
}

/// The colour attachments a render case declares: the single `attachment`
/// field or the MRT `attachments` list, never both.
///
/// *Neither* field is the zero-colour shape (`research/docs/23` §3.3, v46), and
/// it is a well-formed case only when the pass's landing is its stored depth
/// surface: with no colour attachment to observe, that stored surface is the
/// whole observation, and the case has to state its identity and expectation to
/// be comparable. That test is the same one the depth review below makes at
/// length; here it is the admission test that keeps the two absences from being
/// read as an omitted section, so every other case keeps the refusal it had.
private func colorAttachments(_ definition: RenderCaseDefinition) throws -> [RenderAttachmentDefinition] {
    switch (definition.attachment, definition.attachments) {
    case (let single?, nil):
        return [single]
    case (nil, let many?):
        try require(!many.isEmpty, "\(definition.id): the attachment list is empty")
        return many
    case (nil, nil):
        // The zero-colour depth shape: a case with no colour attachment is
        // admitted only when its depth attachment is a stored surface with the
        // identity and the expectation that make the landing observable.
        if let depth = definition.depth, depth.store == "store",
           depth.allocation != nil, depth.view != nil, depth.expected_hex != nil {
            return []
        }
        throw OracleError("\(definition.id): exactly one of attachment and attachments is required")
    case (_?, _?):
        throw OracleError("\(definition.id): attachment and attachments are mutually exclusive")
    }
}

/// The `MTLVertexFormat` one contract format spelling names
/// (`metal_api_core::provider::VertexFormat`), or `nil` for a spelling this
/// oracle does not build a descriptor from.
private func vertexFormat(_ spelling: String) -> MTLVertexFormat? {
    switch spelling {
    case "float32x2": return .float2
    case "float32x3": return .float3
    case "float32x4": return .float4
    case "uint32": return .uint
    default: return nil
    }
}

@available(macOS 11.0, *)
private func loadRenderSource(_ pin: RenderSourcePin, root: URL) throws -> Data {
    // The same review rule as the compute pins (`validateSource`): the identity
    // is part of the manual proof, so a re-hashed fixture cannot admit new
    // source for execution.
    let bytes = try readBoundedFile(root.appendingPathComponent(pin.path).standardizedFileURL)
    try require(sha256(bytes) == pin.sha256, "Render shader SHA-256 mismatch: \(pin.path)")
    return bytes
}

@available(macOS 11.0, *)
private func loadRenderCases(_ suite: SuiteDefinition, root: URL) throws -> [ValidatedRender] {
    var renderCases = [ValidatedRender]()
    for definition in suite.render_cases ?? [] {
        renderCases.append(try validateRenderCase(definition, root: root))
    }
    let ids = renderCases.map { $0.definition.id }
    try require(Set(ids).count == ids.count, "\(suite.suite): duplicate render case id")
    let computeIDs = Set(suite.cases.map { $0.id })
    for id in ids {
        try require(!computeIDs.contains(id),
                    "\(suite.suite): render case \(id) repeats a compute case id")
    }
    return renderCases
}

/// The two texels a reviewed instance-tint stream stores (`research/docs/23`
/// §3.3, v31): one `float32x4` record per instance, read from the start of the
/// stream's own bytes exactly as `conformance/compare.py`'s
/// `_instanced_declaration` reads them. Every component is exactly `0.0` or
/// `1.0`, so the byte an 8-bit UNORM attachment stores is exactly `0x00` or
/// `0xff` and the expectation does not have to model a rounding rule
/// (`research/docs/23` §3.5). Any other component is refused, and the two
/// texels have to differ, or the halves could not show which record each
/// instance read.
private func instancedTintTexels(_ stream: ValidatedVertexStream,
                                 id: String) throws -> (left: Data, right: Data) {
    let recordBytes = 16
    try require(stream.bytes.count >= 2 * recordBytes,
                "\(id): the instance tint stream carries two \(recordBytes)-byte records")
    let bytes = stream.bytes
    var texels = [Data]()
    for record in 0..<2 {
        var texel = Data()
        texel.reserveCapacity(4)
        for component in 0..<4 {
            let base = bytes.startIndex + record * recordBytes + component * 4
            let bits = UInt32(bytes[base])
                | (UInt32(bytes[base + 1]) << 8)
                | (UInt32(bytes[base + 2]) << 16)
                | (UInt32(bytes[base + 3]) << 24)
            let value = Float(bitPattern: bits)
            try require(value == 0.0 || value == 1.0,
                        "\(id): instance tint record \(record), component \(component) "
                        + "has to be zero or one")
            texel.append(value == 1.0 ? 0xff : 0x00)
        }
        texels.append(texel)
    }
    try require(texels[0] != texels[1],
                "\(id): the two instance tints have to differ")
    return (left: texels[0], right: texels[1])
}

/// The render milestone's shape, admitted as a whitelist rather than as a
/// per-case table.
///
/// The first render increment has exactly one render shape (`research/docs/23`
/// §1.2, §3), so the shape itself is the review and a fixture cannot widen it by
/// renaming a case. Everything the runtime reads is decoded and checked here,
/// before a device exists, exactly as `validateShape` does for compute.
@available(macOS 11.0, *)
private func validateRenderCase(_ definition: RenderCaseDefinition,
                                root: URL) throws -> ValidatedRender {
    let reviewed = try reviewedModule(for: definition)
    try require(definition.vertex_entry == reviewed.vertex_entry
                && definition.fragment_entry == reviewed.fragment_entry
                && definition.metal == reviewed.metal,
                "\(definition.id): unreviewed render pipeline identity")
    let sourceBytes = try loadRenderSource(reviewed.metal, root: root)
    guard let source = String(data: sourceBytes, encoding: .utf8) else {
        throw OracleError("\(definition.id): reviewed MSL source is not UTF-8")
    }
    // The vertex-input half: the reviewed layout is an identity, not a knob, so
    // a case declares exactly the streams the module was written for, and each
    // binding carries its own bytes (`research/docs/23` §3.6). The footprint and
    // index-value rules mirror `render.rs::plan_vertex_input`: they are proved
    // here, before a device exists, because Metal would read past a short buffer
    // without refusing.
    let vertexStreams: [ValidatedVertexStream]
    let indexStream: ValidatedIndexStream?
    switch (definition.vertex_layout, definition.vertex_buffers, definition.indices) {
    case (nil, nil, nil):
        try require(definition.vertices == 3,
                    "\(definition.id): expected the full-screen triangle")
        // The base vertex only exists for an indexed draw: both APIs add it to
        // the index values, and a non-indexed draw has none to add it to
        // (`research/docs/23` §3.3, v34).
        try require((definition.base_vertex ?? 0) == 0,
                    "\(definition.id): a base vertex needs an index buffer")
        vertexStreams = []
        indexStream = nil
    case (let layout?, let bindings?, let indices?):
        guard let reviewedBuffers = reviewed.buffers else {
            throw OracleError("\(definition.id): a vertex layout selects an unreviewed module")
        }
        try require(layout.buffers == reviewedBuffers,
                    "\(definition.id): the vertex layout is not the reviewed one")
        // The reviewed blend shape (`research/docs/23` §3.3, v40) is the third
        // pair shape: the same two-attribute stream and module as the depth and
        // culling fixtures, this time with the blend state stated and neither a
        // depth attachment nor a culling state. Like
        // `conformance/compare.py::_blend_declaration` and
        // `provider-capture.rs::render_geometry`, the blend rule is classified
        // ahead of the other two, so a case that declares more than one of the
        // three pair shapes is refused by the shape it claims rather than
        // silently read as another fixture.
        if let blend = definition.blend {
            try require(layout.buffers.count == 1
                        && layout.buffers[0].attributes.count == 2,
                        "\(definition.id): the reviewed blend stream is one stride-32 "
                        + "stream with two attributes")
            try require(definition.depth == nil && definition.cull == nil,
                        "\(definition.id): the reviewed blend shape carries neither a depth "
                        + "attachment nor a culling state")
            try require(blend.count == 1,
                        "\(definition.id): the reviewed blend shape states one attachment")
            let state = blend[0]
            try require(state.source_rgb == "source_alpha"
                        && state.destination_rgb == "one_minus_source_alpha"
                        && state.source_alpha == "source_alpha"
                        && state.destination_alpha == "one_minus_source_alpha"
                        && state.operation == "add",
                        "\(definition.id): the reviewed blend state is source alpha against "
                        + "one-minus-source-alpha with an add")
            try require(definition.attachment != nil && definition.attachments == nil,
                        "\(definition.id): the reviewed blend shape is the single-attachment "
                        + "shape")
        }
        // The reviewed cull pair (`research/docs/23` §3.3, v39) shares the
        // depth fixture's module and stream shape, and the culling state is
        // classified ahead of the depth shape, exactly as
        // `conformance/compare.py` and
        // `provider-capture.rs::render_geometry` order the same rules: the pair
        // shapes are mutually exclusive, so a case that declares both is
        // refused by the cull rule rather than silently read as the depth
        // fixture.
        if let cull = definition.cull {
            try require(layout.buffers.count == 1
                        && layout.buffers[0].attributes.count == 2,
                        "\(definition.id): the reviewed cull stream is one stride-32 "
                        + "stream with two attributes")
            try require(cull.mode == "back" && cull.winding == "counter_clockwise",
                        "\(definition.id): the reviewed cull state is back faces with a "
                        + "counter-clockwise front")
            try require(definition.depth == nil,
                        "\(definition.id): the reviewed cull shape carries no depth attachment")
        }
        // The reviewed stencil shape (`research/docs/23` §3.3, v47) is the
        // fourth pair shape: the same two-attribute stream and module as the
        // depth fixture, the culling pair and the blend shape, this time with
        // the rail-owned stencil surface and the reviewed stencil state in
        // place of the depth attachment. Like the two classifications above it
        // is read ahead of the depth rule — a case that declares more than one
        // of the pair shapes is refused by the shape it claims rather than
        // silently read as another fixture.
        if definition.stencil != nil || definition.stencil_test != nil {
            try require(layout.buffers.count == 1
                        && layout.buffers[0].attributes.count == 2,
                        "\(definition.id): the reviewed stencil stream is one stride-32 "
                        + "stream with two attributes")
            // The combined depth-stencil resolve shape (`research/docs/23` §3.3,
            // v60) is the one stencil case that also opens a depth attachment:
            // both surfaces resolve, and the depth resolve is what the
            // `depth_resolved_sample` filter follows. Every other stencil case
            // carries neither a depth attachment nor a culling or blend state.
            if definition.stencil_resolve == nil {
                try require(definition.depth == nil && definition.cull == nil
                            && definition.blend == nil,
                            "\(definition.id): the reviewed stencil shape carries no depth "
                            + "attachment, culling state or blend state")
            } else {
                try require(definition.depth != nil && definition.depth_resolve != nil
                            && definition.cull == nil && definition.blend == nil,
                            "\(definition.id): the combined stencil-resolve shape opens both "
                            + "surfaces with their resolves and no culling or blend state")
            }
        }
        // The reviewed depth shape (`research/docs/23` §3.3, v36) is the
        // two-attribute stream, and no rail was reviewed against drawing it
        // *without* its depth pair: the surface is what the case exists to
        // exercise, exactly as `render.rs::reviewed_depth_geometry` refuses a
        // depth-shaped draw that carries no attachment. The three pair shapes
        // above draw the same stream shape under their own states, so only a
        // cull-less, blend-less, stencil-less case is the depth fixture.
        if definition.cull == nil, definition.blend == nil,
           definition.stencil == nil, definition.stencil_test == nil,
           layout.buffers.count == 1 && layout.buffers[0].attributes.count == 2 {
            try require(definition.depth != nil,
                        "\(definition.id): the reviewed depth shape carries a depth attachment")
        }
        // The draw's vertex offset (`research/docs/23` §3.3, v34): absent means
        // zero, the shape every pre-v34 case draws. It takes part in the
        // footprint proof below, because the vertex a draw reads is
        // `base_vertex + index`.
        let baseVertex = definition.base_vertex ?? 0
        try require(bindings.count == reviewedBuffers.count,
                    "\(definition.id): one binding per reviewed stream")
        // The reviewed indexed fixture draws six `uint16` indices over the four
        // stream vertices, which is what makes the expectation a covered 2x2
        // attachment rather than a partially drawn one. The reviewed blend
        // shape (`research/docs/23` §3.3, v40) is the exception: its one
        // oversize triangle is drawn through three indices over the three
        // vertices its stream carries.
        if definition.blend != nil {
            try require(definition.vertices == 3,
                        "\(definition.id): the reviewed blend draw is the three-index triangle")
        } else {
            try require(definition.vertices == 6,
                        "\(definition.id): expected the reviewed six-index quad")
        }
        var resolved = [ValidatedVertexStream]()
        for (binding, buffer) in bindings.enumerated() {
            try require(buffer.view > 0 && buffer.allocation > 0,
                        "\(definition.id): zero vertex stream identity")
            let bytes = try decodeHex(buffer.initial_hex,
                                      context: "\(definition.id) vertex stream \(binding)")
            try require(UInt64(bytes.count) == buffer.length,
                        "\(definition.id): vertex stream \(binding) declares "
                        + "\(buffer.length) bytes and carries \(bytes.count)")
            resolved.append(ValidatedVertexStream(binding: binding,
                                                  stride: reviewedBuffers[binding].stride,
                                                  step: reviewedBuffers[binding].resolvedStep,
                                                  attributes: reviewedBuffers[binding].attributes,
                                                  offset: buffer.offset,
                                                  bytes: bytes))
        }
        guard let format = ReviewedIndexType(spelling: indices.format) else {
            throw OracleError("\(definition.id): unsupported index format \(indices.format)")
        }
        try require(indices.view > 0 && indices.allocation > 0,
                    "\(definition.id): zero index buffer identity")
        let indexBytes = try decodeHex(indices.initial_hex,
                                       context: "\(definition.id) index buffer")
        try require(UInt64(indexBytes.count) == indices.length,
                    "\(definition.id): the index buffer declares \(indices.length) bytes "
                    + "and carries \(indexBytes.count)")
        let indexCount = definition.vertices
        let indexFootprint = indexCount * format.byteWidth
        try require(UInt64(indexBytes.count) >= indexFootprint,
                    "\(definition.id): the index view holds \(indexBytes.count) bytes, "
                    + "the draw reads \(indexFootprint)")
        // The highest index value decides how many vertices the streams have to
        // cover, the same span `render.rs` plans from the same bytes.
        var span: UInt64 = 0
        indexBytes.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard let base = raw.baseAddress else { return }
            for offset in stride(from: 0, to: Int(indexFootprint), by: Int(format.byteWidth)) {
                let value: UInt64
                switch format {
                case .uint16:
                    value = UInt64(base.load(fromByteOffset: offset, as: UInt16.self))
                case .uint32:
                    value = UInt64(base.load(fromByteOffset: offset, as: UInt32.self))
                }
                span = max(span, value + 1)
            }
        }
        let instanceCount = definition.instance_count ?? 1
        if resolved.contains(where: { $0.step == "per_instance" }) {
            // The reviewed instanced shape runs exactly two instances, one per
            // half of the attachment (`research/docs/23` §3.3, v31): any other
            // count describes a draw no rail has been reviewed against, and
            // the per-half expectation below could not follow it.
            try require(instanceCount == 2,
                        "\(definition.id): the reviewed instanced draw runs exactly two "
                        + "instances")
        }
        // The offset takes part in the proof: a span that fits on its own can
        // still reach past the stream once the base vertex is added
        // (`research/docs/23` §3.3, v34). The proof only has to decide whether
        // a stream covers the span, so an unrepresentable sum saturates — it is
        // by definition larger than any buffer this oracle admits, the same
        // answer `render.rs::plan_vertex_input` computes with `saturating_add` —
        // instead of trapping on the addition.
        let (spanAndOffset, offsetOverflowed) = span.addingReportingOverflow(baseVertex)
        let requiredSpan = offsetOverflowed ? UInt64.max : spanAndOffset
        // A per-instance stream advances once per instance instead of once per
        // index-selected vertex, so the draw reads `instanceCount` records of
        // it (`research/docs/23` §3.3, v31); every other stream has to cover
        // the span the indices reach, exactly as before.
        for stream in resolved {
            let covered = UInt64(stream.bytes.count) / stream.stride
            if stream.step == "per_instance" {
                try require(instanceCount <= covered,
                            "\(definition.id): the draw reads \(instanceCount) records of "
                            + "binding \(stream.binding), which covers \(covered)")
            } else {
                try require(requiredSpan <= covered,
                            "\(definition.id): index values reach vertex \(requiredSpan - 1) of "
                            + "binding \(stream.binding) through base vertex \(baseVertex), "
                            + "which covers \(covered)")
            }
        }
        vertexStreams = resolved
        indexStream = ValidatedIndexStream(format: format, indexCount: indexCount,
                                           offset: indices.offset,
                                           baseVertex: baseVertex,
                                           bytes: indexBytes)
    default:
        throw OracleError("\(definition.id): a vertex layout, its bindings and the index "
                          + "buffer are declared together")
    }
    let attachments = try colorAttachments(definition)
    // Whether the pass's landing is its stored depth attachment is what the two
    // colour rules below read, so the depth section's store action is consulted
    // before the attachments are classified. The depth review at the end of
    // this function is what pins that action's spelling, identity and
    // expectation — a half-declared store is refused there — which is the same
    // split `conformance/compare.py` makes when it parses the depth declaration
    // ahead of the attachment list and then states the two rules
    // (`research/docs/23` §3.3, v43/v45).
    let depthLanding = definition.depth?.store != nil
    // One expectation per attachment, in location order: the single form
    // carries it at the case level, the MRT form on each attachment entry. A
    // discarded attachment carries none at all — its bytes disappear from the
    // observable surface, so there is nothing to compare (`research/docs/23`
    // §3.6, v19). The single-attachment discard is the v45 depth-only shape,
    // and it is the one single-attachment case whose expectation is absent: the
    // pass drops the colour bytes and the depth surface its own section names
    // is the whole observation. A single-attachment discard without that
    // landing, and a depth-only shape that still spells a case-level
    // expectation, are both refused — the wording the MRT discard arm uses
    // (`conformance/compare.py` reports the same two refusals the same way).
    // The v46 zero-colour shape carries no expectation either: the pass has no
    // colour attachment at all, so a case-level expectation would claim bytes
    // no surface holds, and the arm below is where that reading is stated.
    let expectedHexes: [String?]
    if attachments.isEmpty {
        // The zero-colour shape (`research/docs/23` §3.3, v46): the pass carries
        // no colour attachment, so there is no colour landing an expectation
        // could describe. A case-level `expected_hex` would claim bytes no
        // surface holds, so it is refused with the same reading the v45 discard
        // gets — the stored depth surface its own section names is the whole
        // observation, and the depth review below is what pins its texels.
        try require(definition.expected_hex == nil,
                    "\(definition.id): a pass without a colour attachment carries no expectation")
        expectedHexes = []
    } else if definition.attachment != nil {
        let discards = attachments[0].store == "dontcare"
        if let top = definition.expected_hex {
            try require(!discards,
                        "\(definition.id): a discarded attachment carries no expectation")
            try require(attachments[0].expected_hex == nil,
                        "\(definition.id): a single attachment carries no expected_hex")
            expectedHexes = [top]
        } else if discards {
            try require(depthLanding,
                        "\(definition.id): a discarded attachment carries no expectation")
            // The single form states its expectation at the case level, so an
            // entry-level one is the MRT spelling and stays refused here, the
            // same fields rule the MRT branch of the comparison states.
            try require(attachments[0].expected_hex == nil,
                        "\(definition.id): a single attachment carries no expected_hex")
            expectedHexes = [nil]
        } else {
            throw OracleError("\(definition.id): a single attachment needs expected_hex")
        }
    } else {
        try require(definition.expected_hex == nil,
                    "\(definition.id): an attachment list carries its own expected_hex")
        expectedHexes = attachments.map { attachment in attachment.expected_hex }
    }
    // The coverage claim (`research/docs/23` §3.3, v38): only the
    // single-attachment shape may make it, and only the one spelling exists —
    // `"partial"`, meaning the pass's draw covers part of the attachment, so
    // the expectation mixes the fragment output with the colour the clear load
    // started from. `conformance/compare.py` and the Rust providers read the
    // same field the same way, and the default stays the milestone's stricter
    // rule because a case that says nothing claims every texel.
    if let coverage = definition.coverage {
        try require(coverage == "partial",
                    "\(definition.id): the only coverage claim is \"partial\"")
        try require(definition.attachment != nil,
                    "\(definition.id): the coverage claim is the single-attachment shape")
    }
    // The pass-wide multisample raster (`research/docs/23` §3.3, v51): the
    // first increment reviews one shape — a single colour attachment opened
    // from a clear, four samples, no depth or stencil surface, no present
    // action, no ICB and no wildcard texels — and its expectation follows the
    // resolve rule below instead of the coverage rule. `conformance/compare.py`
    // and the Rust providers read the same field the same way.
    if let multisample = definition.multisample {
        try require(definition.attachment != nil,
                    "\(definition.id): the multisample raster is the single-attachment shape")
        try require(multisample.sample_count == 4,
                    "\(definition.id): the reviewed multisample raster is four samples")
        // The raster's own expectation shape depends on what it opens
        // (`research/docs/23` §3.3, v51/v53): a colour-only raster resolves
        // fragment output and clear into partial texels and therefore has to
        // claim partial coverage, while a raster that opens a rail-owned depth
        // surface is the depth fixture's own shape — both of its primitives
        // cover every sample, the near one wins and every texel is one fragment
        // output.
        if let depth = definition.depth {
            // A stored multisampled depth surface is admitted from v57 on,
            // through the resolve the case then has to state: its texels are
            // only observable as the resolve's reduction, so a stored surface
            // without one is refused, and a resolve beside a discarded surface
            // is refused too — it is the stored surface's own tail
            // (`research/docs/23` §3.3, v57).
            if depth.store != nil {
                try require(definition.depth_resolve != nil,
                            "\(definition.id): a stored multisampled depth surface needs its "
                            + "depth resolve")
                if let filter = definition.depth_resolve?.filter {
                    try require(filter == "sample0" || filter == "min" || filter == "max",
                                "\(definition.id): unsupported depth resolve filter \(filter)")
                }
            } else {
                try require(definition.depth_resolve == nil,
                            "\(definition.id): a depth resolve needs a stored depth surface")
            }
            // The combined shape's stencil half states the same stored
            // surface rule its single-surface sibling does
            // (`research/docs/23` §3.3, v60).
            if let stencil = definition.stencil {
                if stencil.store != nil {
                    try require(definition.stencil_resolve != nil,
                                "\(definition.id): a stored multisampled stencil surface needs "
                                + "its stencil resolve")
                } else {
                    try require(definition.stencil_resolve == nil,
                                "\(definition.id): a stencil resolve needs a stored stencil "
                                + "surface")
                }
            }
            try require(definition.coverage == nil,
                        "\(definition.id): a multisample pass with a depth surface claims no "
                        + "partial coverage")
        } else if let stencil = definition.stencil {
            if stencil.store != nil {
                try require(definition.stencil_resolve != nil,
                            "\(definition.id): a stored multisampled stencil surface needs its "
                            + "stencil resolve")
                if let filter = definition.stencil_resolve?.filter {
                    try require(filter == "sample0" || filter == "depth_resolved_sample",
                                "\(definition.id): unsupported stencil resolve filter \(filter)")
                }
            } else {
                try require(definition.stencil_resolve == nil,
                            "\(definition.id): a stencil resolve needs a stored stencil surface")
            }
            try require(definition.coverage == nil,
                        "\(definition.id): a multisample pass with a stencil surface claims no "
                        + "partial coverage")
        } else if definition.depth_resolve != nil {
            throw OracleError("\(definition.id): a depth resolve needs a stored depth surface")
        } else {
            try require(definition.coverage == "partial",
                        "\(definition.id): the multisample raster has to claim partial coverage")
        }
        // The depthResolvedSample filter names the sample the depth resolve
        // selected, so a case that states it without one is refused
        // (`research/docs/23` §3.3, v60).
        if definition.stencil_resolve?.filter == "depth_resolved_sample" {
            try require(definition.depth_resolve != nil,
                        "\(definition.id): the depth_resolved_sample stencil resolve names "
                        + "the sample the depth resolve selects, so the case has to state a "
                        + "depth resolve")
        }
        // The device gate (`research/docs/23` §3.3, v57d): a case that
        // requires a filter has to state the resolve whose filter it names —
        // the gate is the case's own admission condition, not a second
        // spelling that could drift — and only the two filters a device may
        // lack are gateable.
        if let gate = definition.requires_depth_resolve_filter {
            try require(gate == "min" || gate == "max",
                        "\(definition.id): the device gate names the min or max depth "
                        + "resolve filter")
            try require(definition.depth_resolve?.filter == gate,
                        "\(definition.id): the device gate has to name the resolve filter "
                        + "the case states")
        }
        try require(definition.depth == nil || definition.stencil == nil
                    || definition.stencil_resolve != nil,
                    "\(definition.id): the multisample raster opens one depth-stencil "
                    + "surface: a combined surface is admitted only through a stencil "
                    + "resolve")
        try require(definition.wildcard_texels == nil,
                    "\(definition.id): the multisample raster claims every texel it resolves")
        // The reviewed multisample shapes carry no vertex offset, and the
        // oracle's own footprint proof is the only gate that would notice one
        // (`research/docs/23` §3.3, v54 review H1); the fixture gate states the
        // same rule, as does `conformance/compare.py`.
        try require(definition.base_vertex == nil || definition.base_vertex == 0,
                    "\(definition.id): the reviewed multisample shapes carry no base vertex")
        // The trace contract's own rule one line up: a multisampled pass opens
        // its attachment from a clear, which `conformance/compare.py` states
        // explicitly (`research/docs/23` §3.3, v54 review L1).
        try require(definition.attachment?.load == "clear",
                    "\(definition.id): the reviewed multisample pass opens its attachment "
                    + "from a clear")
    }
    // The wildcard channel (`research/docs/23` §3.3, v33): a case may name the
    // texels whose bytes it does not claim, and the undefined pre-pass contents
    // of a `dontcare` load are exactly what makes an unclaimed byte
    // legitimate. The list is the single-attachment shape's, it has to name at
    // least one texel and leave at least one observed, and every entry has to
    // name a texel of that attachment exactly once. An absent list means the
    // case claims every texel, the semantics every case before v33 has.
    if let wildcards = definition.wildcard_texels {
        try require(definition.attachment != nil,
                    "\(definition.id): the wildcard channel is the single-attachment shape")
        try require(!wildcards.isEmpty,
                    "\(definition.id): a wildcard list has to name at least one texel")
        var seen = Set<Int>()
        for texel in wildcards {
            try require(texel >= 0,
                        "\(definition.id): wildcard texel \(texel) is outside the attachment")
            try require(!seen.contains(texel),
                        "\(definition.id): duplicate wildcard texel \(texel)")
            seen.insert(texel)
        }
        // The single-attachment shape was proved above, so `attachments` holds
        // exactly the one entry the list describes: only undefined contents may
        // leave texels unclaimed, the list has to leave at least one texel
        // observed, and every entry has to name a texel of the attachment.
        let attachment = attachments[0]
        try require(attachment.load == "dontcare",
                    "\(definition.id): only a dontcare load may leave texels unclaimed")
        let texelCount = attachment.width * attachment.height
        try require(wildcards.count < texelCount,
                    "\(definition.id): a wildcard list has to leave at least one texel observed")
        for texel in wildcards {
            try require(texel < texelCount,
                        "\(definition.id): wildcard texel \(texel) is outside the attachment")
        }
    }
    // The v19 pass-level rule core admission states as
    // `AllRenderAttachmentsDiscarded`: at least one attachment has to stay on
    // the observable surface, or "nothing landed" would pass as "landed
    // correctly". A stored depth attachment is a landing too (`research/docs/23`
    // §3.3, v43/v45): the depth-only shape discards every colour attachment and
    // keeps the depth surface, and the depth texels are then the whole
    // comparison, the alternative `conformance/compare.py` states once its
    // depth declaration is parsed. The depth review below is what pins the
    // identity and the expectation that make that landing observable.
    try require(attachments.contains { $0.store == "store" } || depthLanding,
                "\(definition.id): every colour attachment discards, leaving no observable landing point")
    var validatedAttachments = [ValidatedRenderAttachment]()
    for (index, attachment) in attachments.enumerated() {
        try require(attachment.format == "rgba8_unorm"
                    || attachment.format == "bgra8_unorm"
                    || attachment.format == "r32float",
                    "\(definition.id): unsupported attachment format")
        try require(attachment.width >= 1 && attachment.width <= 4
                    && attachment.height >= 1 && attachment.height <= 4,
                    "\(definition.id): the attachment extent is one to four texels per axis")
        try require(attachment.allocation > 0 && attachment.view > 0,
                    "\(definition.id): zero attachment identity")
        try require(attachment.store == "store" || attachment.store == "dontcare",
                    "\(definition.id): unsupported attachment store op \(attachment.store)")
        let stored = attachment.store == "store"
        try require(definition.viewport == [0, 0, UInt64(attachment.width), UInt64(attachment.height)],
                    "\(definition.id): the viewport must cover the attachment")
        if let scissor = definition.scissor {
            try require(scissor.count == 4,
                        "\(definition.id): a scissor is four numbers")
            try require(scissor[2] > 0 && scissor[3] > 0
                        && scissor[0] + scissor[2] <= UInt64(attachment.width)
                        && scissor[1] + scissor[3] <= UInt64(attachment.height),
                        "\(definition.id): a scissor has to be a non-empty rectangle "
                        + "inside the attachment")
        }
        let byteCount = attachment.width * attachment.height * 4
        // A stored attachment carries the whole expectation and the byte-level
        // review it makes possible; a discarded attachment carries none, and
        // an expectation arriving for one is refused (`research/docs/23` §3.6,
        // v19).
        let expected: Data?
        if let hex = expectedHexes[index] {
            try require(stored,
                        "\(definition.id): a discarded attachment carries no expected_hex")
            let texels = try decodeHex(hex, context: "\(definition.id) expected texels")
            try require(texels.count == byteCount,
                        "\(definition.id): expected texel bytes do not match the attachment")
            // What a drawn texel has to be depends on what the pass started
            // from. A clearing pass has nothing to preserve, so every texel
            // has to be the same fragment output (`research/docs/23` §1.3) —
            // a partially covered attachment cannot be asserted as correct
            // unless the case claims that shape, which the coverage claim below
            // is. A loading pass deliberately keeps the bytes it was handed
            // wherever the draw missed, so its expectation is classified once
            // the previous bytes are decoded, below.
            let texel = Data(texels.prefix(4))
            var texelCount = 0
            // The resolve rule covers the colour-only raster and the combined
            // depth-stencil shape whose stencil half resolves (`research/docs/23`
            // §3.3, v51/v60); the single-surface masked rasters keep the pair's
            // uniform rule below.
            if attachment.load == "clear", let multisample = definition.multisample,
               (definition.depth == nil && definition.stencil == nil
                || definition.stencil_resolve != nil) {
                // The multisample resolve (`research/docs/23` §3.3, v51): every
                // texel is the arithmetic mean of the samples a primitive
                // covered, so the expectation has to be a k-of-`sample_count`
                // mix of the fragment output and the clear colour — and both
                // extremes and at least one partial mix have to appear, or the
                // fixture would claim a pattern a single-sample raster could
                // produce. A mix that is not exactly representable is refused:
                // the fixture has to choose colours whose mixes divide
                // exactly, which is what keeps the expectation independent of
                // a driver's rounding rule.
                guard let clearHex = attachment.clear_hex else {
                    throw OracleError("\(definition.id): a clear attachment needs clear_hex")
                }
                let clearBytes = try decodeHex(clearHex, context: "\(definition.id) clear colour")
                try require(clearBytes.count == 4,
                            "\(definition.id): a clear colour is four bytes")
                let samples = Int(multisample.sample_count)
                var coveredSeen = Set<Int>()
                for offset in stride(from: 0, to: texels.count, by: 4) {
                    let chunk = Data(texels[offset..<(offset + 4)])
                    var matched: Int?
                    for covered in 0...samples {
                        if chunk == resolveTexel(fragment: texel, clear: clearBytes,
                                                 covered: covered, samples: samples) {
                            matched = covered
                            break
                        }
                    }
                    guard let covered = matched else {
                        throw OracleError("\(definition.id): texel \(offset / 4) is not the "
                                          + "resolve of any coverage of the \(samples)-sample "
                                          + "raster")
                    }
                    coveredSeen.insert(covered)
                    texelCount += 1
                }
                try require(coveredSeen.contains { $0 > 0 && $0 < samples },
                            "\(definition.id): a multisample expectation needs at least one "
                            + "partially covered texel")
                try require(coveredSeen.contains(0) && coveredSeen.contains(samples),
                            "\(definition.id): a multisample expectation needs both a fully "
                            + "covered and an uncovered texel")
            } else if attachment.load == "clear", definition.coverage == "partial" {
                // The coverage claim (`research/docs/23` §3.3, v38): the draw
                // covers part of the attachment, so every texel is either the
                // fragment output or the colour the clear load started it
                // from, and both have to appear — a fixture that claimed
                // "everything" or "nothing" could not show the partial coverage
                // it declares. The claim stands in place of the scissor and
                // instanced branches below, the same precedence
                // `conformance/compare.py` and the Rust providers read.
                guard let clearHex = attachment.clear_hex else {
                    throw OracleError("\(definition.id): a clear attachment needs clear_hex")
                }
                let clearBytes = try decodeHex(clearHex, context: "\(definition.id) clear colour")
                try require(clearBytes.count == 4,
                            "\(definition.id): a clear colour is four bytes")
                var drawn = 0
                var kept = 0
                for offset in stride(from: 0, to: texels.count, by: 4) {
                    let chunk = Data(texels[offset..<(offset + 4)])
                    if chunk == texel {
                        drawn += 1
                    } else if chunk == clearBytes {
                        kept += 1
                    } else {
                        throw OracleError("\(definition.id): texel \(offset / 4) of a partial "
                                          + "coverage claim has to be the fragment output or "
                                          + "the clear colour")
                    }
                    texelCount += 1
                }
                try require(drawn > 0 && kept > 0,
                            "\(definition.id): a partial coverage claim needs both drawn "
                            + "and clear texels")
            } else if attachment.load == "clear" {
                // A scissored pass covers a known rectangle: inside it every
                // texel is the fragment output and outside it every texel is the
                // clear colour (`research/docs/23` §3.3, v29). Both halves have
                // to appear, or the fixture could not show the clip ran.
                //
                // The reviewed instanced pass (`research/docs/23` §3.3, v31)
                // declares no scissor, and its expectation is per half instead:
                // each instance's tint covers the half its `instance_id` shift
                // puts it on, so the left half has to carry the first record's
                // texel and the right half the second's. A uniform expectation
                // could not show that the per-instance stream stepped at all,
                // and a swapped pair would read as agreement on the wrong
                // halves — which is why the two records have to differ and
                // neither may equal the clear colour. The scissor branch keeps
                // its precedence, mirroring `conformance/compare.py`.
                var instanceTexels: (left: Data, right: Data)?
                if definition.scissor == nil,
                   let tintStream = vertexStreams.first(where: { $0.step == "per_instance" }) {
                    let (left, right) = try instancedTintTexels(tintStream, id: definition.id)
                    guard let clearHex = attachment.clear_hex else {
                        throw OracleError("\(definition.id): a clear attachment needs clear_hex")
                    }
                    let clearBytes = try decodeHex(clearHex, context: "\(definition.id) clear colour")
                    try require(clearBytes.count == 4,
                                "\(definition.id): a clear colour is four bytes")
                    try require(left != clearBytes && right != clearBytes,
                                "\(definition.id): the two instance tints have to differ "
                                + "from the clear colour")
                    instanceTexels = (left: left, right: right)
                }
                var scissorClear: Data?
                var covered = 0
                if let scissor = definition.scissor {
                    let clearHex = attachment.clear_hex ?? ""
                    scissorClear = try decodeHex(clearHex, context: "\(definition.id) clear colour")
                    _ = scissor
                }
                for offset in stride(from: 0, to: texels.count, by: 4) {
                    if let scissor = definition.scissor, let clearBytes = scissorClear {
                        let index = offset / 4
                        let column = UInt64(index % attachment.width)
                        let row = UInt64(index / attachment.width)
                        let inside = column >= scissor[0] && column < scissor[0] + scissor[2]
                            && row >= scissor[1] && row < scissor[1] + scissor[3]
                        let expected = inside ? texel : clearBytes
                        try require(Data(texels[offset..<(offset + 4)]) == expected,
                                    "\(definition.id): texel \(index) has to follow the declared scissor")
                        covered += inside ? 1 : 0
                        texelCount += 1
                        continue
                    }
                    if let tints = instanceTexels {
                        let index = offset / 4
                        let column = index % attachment.width
                        let half = column < attachment.width / 2 ? tints.left : tints.right
                        try require(Data(texels[offset..<(offset + 4)]) == half,
                                    "\(definition.id): texel \(index) has to carry the "
                                    + "instance tint of its half")
                        texelCount += 1
                        continue
                    }
                    try require(Data(texels[offset..<(offset + 4)]) == texel,
                                "\(definition.id): the milestone expects every texel to equal the fragment output")
                    texelCount += 1
                }
                if definition.scissor != nil {
                    try require(covered > 0 && covered < texelCount,
                                "\(definition.id): the scissor has to clip part of the attachment")
                }
            } else {
                texelCount = texels.count / 4
            }
            try require(texelCount == attachment.width * attachment.height,
                        "\(definition.id): attachment texel count mismatch")
            expected = texels
        } else {
            try require(!stored,
                        "\(definition.id): attachment \(attachment.view) needs expected_hex")
            expected = nil
        }
        let clearComponents: [Double]
        let initial: Data?
        switch attachment.load {
        case "clear":
            guard let clearHex = attachment.clear_hex else {
                throw OracleError("\(definition.id): a clear attachment needs clear_hex")
            }
            let clearBytes = try decodeHex(clearHex, context: "\(definition.id) clear colour")
            try require(clearBytes.count == 4, "\(definition.id): a clear colour is four bytes")
            try require(attachment.initial_hex == nil,
                        "\(definition.id): a cleared attachment carries no initial bytes")
            clearComponents = [Double(clearBytes[0]) / 255.0, Double(clearBytes[1]) / 255.0,
                               Double(clearBytes[2]) / 255.0, Double(clearBytes[3]) / 255.0]
            // The sentinel has to be distinguishable from the fragment output,
            // or a pass that never ran would satisfy the expectation. A
            // discarded attachment has no expectation, so there is nothing to
            // distinguish.
            if let expected {
                try require(clearBytes != Data(expected.prefix(4)),
                            "\(definition.id): the clear colour equals the expected texel")
            }
            initial = nil
        case "load":
            guard let initialHex = attachment.initial_hex else {
                throw OracleError("\(definition.id): a loaded attachment needs its previous texels")
            }
            let previous = try decodeHex(initialHex, context: "\(definition.id) initial texels")
            try require(previous.count == byteCount,
                        "\(definition.id): initial texels do not match the attachment")
            clearComponents = []
            initial = previous
            if let expected {
                try require(previous != expected,
                            "\(definition.id): the initial texels equal the expectation")
                // Partial coverage, in both directions: every texel is either
                // the byte the load handed it or the pass's fragment output,
                // every drawn texel carries the *same* output, and both halves
                // appear (`docs/23` §3.3).
                var drawn: Data? = nil
                var drawnCount = 0
                for offset in stride(from: 0, to: expected.count, by: 4) {
                    let chunk = Data(expected[offset..<(offset + 4)])
                    let previousChunk = Data(previous[offset..<(offset + 4)])
                    if chunk == previousChunk {
                        continue
                    }
                    if let drawn {
                        try require(chunk == drawn,
                                    "\(definition.id): drawn texels disagree about the fragment output")
                    } else {
                        drawn = chunk
                    }
                    drawnCount += 1
                }
                // The suite comparator additionally requires at least one
                // *kept* texel, because a loading case whose draw covers
                // everything cannot show that the load happened
                // (`conformance/compare.py`). This oracle's own self-test
                // fixtures are deliberately that shape — the present self-test
                // exists to show the sentinel was replaced, not to falsify the
                // load — so the oracle only insists that something was drawn
                // here and leaves the falsifiability rule to the comparator and
                // to the suite fixtures.
                try require(drawnCount > 0,
                            "\(definition.id): a loaded attachment needs at least one drawn texel")
            }
        case "dontcare":
            // Undefined pre-pass contents (`docs/23` §13, v20): the pass
            // starts from nothing, so neither a clear colour nor initial bytes
            // travel with the attachment, and the texture below is created
            // without any pre-seed. The byte-level "the declared view's bytes
            // differ from the expectation" rule is the suite comparator's; the
            // oracle only has to refuse the two carried-value spellings.
            try require(attachment.clear_hex == nil,
                        "\(definition.id): a dontcare load carries no clear colour")
            try require(attachment.initial_hex == nil,
                        "\(definition.id): a dontcare load carries no initial bytes")
            // The expectation follows the fragment output on every texel the
            // case *claims*; the unclaimed ones are the wildcard list's
            // (`research/docs/23` §3.3, v33). Without a list the v20 rule — and
            // its exact message — stays: the fixture then has no way to say
            // which bytes it does not claim, so it claims all of them, exactly
            // as `conformance/compare.py` reads the same shape.
            if let expected {
                let fragment = Data(expected.prefix(4))
                if let wildcards = definition.wildcard_texels {
                    for texel in 0..<(expected.count / 4) where !wildcards.contains(texel) {
                        try require(Data(expected[(texel * 4)..<(texel * 4 + 4)]) == fragment,
                                    "\(definition.id): texel \(texel) of a dontcare load has to "
                                    + "be the fragment output")
                    }
                } else {
                    for offset in stride(from: 0, to: expected.count, by: 4) {
                        try require(Data(expected[offset..<(offset + 4)]) == fragment,
                                    "\(definition.id): every texel of a dontcare load has to "
                                    + "be the fragment output")
                    }
                }
            }
            clearComponents = []
            initial = nil
        default:
            throw OracleError("\(definition.id): unsupported attachment load op \(attachment.load)")
        }
        // The wildcard mask this attachment carries (`research/docs/23` §3.3,
        // v33): the four bytes of every texel the case leaves unclaimed, stated
        // as offsets inside the attachment's own image — the same image the
        // readback compares, whose first texel sits at offset 0. Only the
        // single-attachment shape may name a list, so every other attachment
        // carries the empty set.
        var wildcardBytes = Set<Int>()
        for texel in definition.wildcard_texels ?? [] {
            wildcardBytes.formUnion((texel * 4)..<(texel * 4 + 4))
        }
        let pixelFormat: MTLPixelFormat
        switch attachment.format {
        case "bgra8_unorm": pixelFormat = .bgra8Unorm
        case "r32float": pixelFormat = .r32Float
        default: pixelFormat = .rgba8Unorm
        }
        validatedAttachments.append(ValidatedRenderAttachment(
            allocation: attachment.allocation, view: attachment.view,
            width: attachment.width, height: attachment.height,
            load: attachment.load, store: attachment.store,
            clearComponents: clearComponents, initial: initial, expected: expected,
            wildcardBytes: wildcardBytes, pixelFormat: pixelFormat))
    }
    // The two reviewed MRT locations write two different byte strings, so a
    // cleared dual case whose locations read back the same texel could not
    // show that both outputs landed (`4080c0ff` vs `ff8040c0`). A discarded
    // location has no expectation, so it takes no part in the comparison.
    if validatedAttachments.count > 1,
       validatedAttachments[0].load == "clear",
       validatedAttachments[1].load == "clear",
       let first = validatedAttachments[0].expected,
       let second = validatedAttachments[1].expected,
       first.prefix(4) == second.prefix(4) {
        throw OracleError("\(definition.id): the two locations read back the same texel")
    }
    // The depth pair (`research/docs/23` §3.3, v36) is part of the reviewed
    // shape: one `depth32float` attachment cleared to one, covering the same
    // render area as the colour attachment, and a `less` test with writes on.
    // Anything else describes a shape no rail has been reviewed against, and a
    // state without its attachment — or the reverse — is refused, mirroring
    // `render.rs::case_depth`.
    try require(definition.depth != nil || definition.depth_test == nil,
                "\(definition.id): a depth test needs a depth attachment")
    let validatedDepth: ValidatedDepth?
    if let depth = definition.depth {
        try require(depth.format == "depth32float",
                    "\(definition.id): the reviewed depth attachment is a depth32float")
        try require(depth.load == "clear",
                    "\(definition.id): the reviewed depth attachment is cleared")
        // The depth pair clears to one; the combined depth-stencil shape
        // clears to 0.7, which is the value its two resolves have to pick
        // between (`research/docs/23` §3.3, v60). Any other clear describes a
        // shape no rail was reviewed against.
        let reviewedClear = definition.stencil_resolve != nil ? 0.7 : 1.0
        guard let clearDepth = depth.clear_depth, clearDepth == reviewedClear else {
            throw OracleError(
                "\(definition.id): the reviewed depth clear is \(reviewedClear)")
        }
        // The depth attachment is a second raster with the pass's own extent
        // (`research/docs/23` §3.3, v36): the two have to agree, the same rule
        // the viewport states for the colour side. The zero-colour shape
        // (`§3.3`, v46) has no colour attachment to state that extent, so the
        // depth surface *is* the pass's render area there: the case's viewport
        // has to cover it, and the scissor, when the case declares one, has to
        // stay inside it — the rectangle rule the colour side states once per
        // attachment.
        if let colour = attachments.first {
            try require(depth.width == colour.width && depth.height == colour.height,
                        "\(definition.id): the depth attachment has to match the colour extent")
        } else {
            try require(definition.viewport == [0, 0, UInt64(depth.width), UInt64(depth.height)],
                        "\(definition.id): the viewport must cover the depth attachment")
            if let scissor = definition.scissor {
                try require(scissor.count == 4,
                            "\(definition.id): a scissor is four numbers")
                try require(scissor[2] > 0 && scissor[3] > 0
                            && scissor[0] + scissor[2] <= UInt64(depth.width)
                            && scissor[1] + scissor[3] <= UInt64(depth.height),
                            "\(definition.id): a scissor has to be a non-empty rectangle "
                            + "inside the depth attachment")
            }
        }
        guard let test = definition.depth_test else {
            throw OracleError("\(definition.id): the reviewed depth shape carries a depth test")
        }
        try require(test.compare == "less" && test.write,
                    "\(definition.id): the reviewed depth state is a less test with writes on")
        // The depth store pair (`research/docs/23` §3.3, v43) is all-or-
        // nothing, mirroring the contract's `DepthStoreIdentityMismatch`: the
        // rail-owned shape every pre-v43 case declares states none of the four
        // fields, and a storing surface states the action, its identity and
        // what the readback has to show. An expectation that equals the clear
        // image is refused because it could not tell "the pass stored the
        // depth texels" from "it never wrote depth" (`docs/23` §3.3, v43).
        let depthStore: ValidatedDepthStore?
        if let spelling = depth.store {
            try require(spelling == "store",
                        "\(definition.id): the only stored depth spelling is \"store\"")
            guard let allocation = depth.allocation,
                  let view = depth.view,
                  let expectedHex = depth.expected_hex else {
                throw OracleError("\(definition.id): a stored depth attachment needs its store "
                                  + "action, its identity and an expectation")
            }
            try require(allocation > 0 && view > 0,
                        "\(definition.id): zero depth attachment identity")
            let expected = try decodeHex(expectedHex,
                                         context: "\(definition.id) expected depth texels")
            try require(expected.count == depth.width * depth.height * 4,
                        "\(definition.id): expected depth texel bytes do not match the depth "
                        + "attachment")
            // The clear image the surface starts from (`float32` 1.0 is
            // `0000803f` in memory order, once per texel) is exactly what a
            // pass that never stored its depth would read back.
            var clearImage = Data()
            clearImage.reserveCapacity(depth.width * depth.height * 4)
            for _ in 0..<(depth.width * depth.height) {
                clearImage.append(contentsOf: [0x00, 0x00, 0x80, 0x3f])
            }
            try require(expected != clearImage,
                        "\(definition.id): the expected depth texels equal the clear depth")
            // The depth landing is a resource of its own: a colour attachment
            // already names these identities, and sharing one would make the
            // colour readback and the depth readback the same landing
            // (`research/docs/23` §3.3, v43).
            for attachment in attachments {
                try require(allocation != attachment.allocation && view != attachment.view,
                            "\(definition.id): the depth attachment reuses the colour "
                            + "attachment's identity")
            }
            depthStore = ValidatedDepthStore(allocation: allocation, view: view,
                                             expected: expected)
        } else {
            try require(depth.allocation == nil && depth.view == nil && depth.expected_hex == nil,
                        "\(definition.id): a discarded depth attachment carries no identity "
                        + "or expectation")
            depthStore = nil
        }
        validatedDepth = ValidatedDepth(width: depth.width, height: depth.height,
                                        clearDepth: clearDepth,
                                        isLess: test.compare == "less", write: test.write,
                                        store: depthStore)
    } else {
        validatedDepth = nil
    }
    // The stencil pair (`research/docs/23` §3.3, v47) is the second reviewed
    // pair-with-state shape: one rail-owned `stencil8` attachment cleared to
    // zero, covering the same render area as the colour attachment, and the
    // reviewed stencil state. The stored values are the whole point of the
    // fixture — the first primitive of the draw writes the value the second
    // one is then tested against — so anything else describes a shape no rail
    // has been reviewed against, and a state without its attachment — or the
    // reverse — is refused, mirroring the two depth rules above.
    try require(definition.stencil != nil || definition.stencil_test == nil,
                "\(definition.id): a stencil test needs a stencil attachment")
    let validatedStencil: ValidatedStencil?
    if let stencil = definition.stencil {
        try require(stencil.format == "stencil8",
                    "\(definition.id): the reviewed stencil attachment is a stencil8")
        try require(stencil.load == "clear",
                    "\(definition.id): the reviewed stencil attachment is cleared")
        guard let clearValue = stencil.clear_value, clearValue == 0 else {
            throw OracleError("\(definition.id): the reviewed stencil clear is 0")
        }
        // The stencil attachment is a second raster with the pass's own extent,
        // exactly as the depth sibling is (`research/docs/23` §3.3, v36): the
        // two have to agree. The colour side's viewport rule already pins
        // every colour attachment to one extent, so the first one is the whole
        // comparison. Only the colour-carrying shape reaches this review: a
        // pass with no colour attachment at all is the v46 zero-colour shape,
        // whose stored depth attachment the classification above refuses
        // beside a stencil pair.
        guard let colour = attachments.first else {
            throw OracleError("\(definition.id): the reviewed stencil shape carries a "
                              + "colour attachment")
        }
        try require(stencil.width == colour.width && stencil.height == colour.height,
                    "\(definition.id): the stencil attachment has to match the colour extent")
        guard let test = definition.stencil_test else {
            throw OracleError("\(definition.id): the reviewed stencil shape carries a stencil test")
        }
        // The reviewed stencil states (`research/docs/23` §3.3, v47/v60): the
        // v47 equal-zero test whose pass op increments-wraps, and the v60
        // combined shape's always test that increments on depth pass and keeps
        // on depth fail — both with both masks wide open. The two states make
        // their fixtures' two primitives differ: the first passes against the
        // cleared zero and writes one while the second drops, and the combined
        // pair splits through the depth test.
        try require((test.compare == "equal" || test.compare == "always")
                    && test.reference == 0
                    && test.read_mask == 255
                    && test.write_mask == 255
                    && test.fail_op == "keep"
                    && test.depth_fail_op == "keep"
                    && test.pass_op == "increment_wrap",
                    "\(definition.id): the reviewed stencil state is the equal-zero test or "
                    + "the combined shape's always test, both with both masks wide open, "
                    + "keeping both failure outcomes and incrementing-wrapping on pass")
        // The stencil store pair (`research/docs/23` §3.3, v49) is all-or-
        // nothing, mirroring the contract's `StencilStoreIdentityMismatch` and
        // the depth sibling's own rule: the rail-owned shape every pre-v49 case
        // declares states none of the four fields, and a storing surface states
        // the action, its identity and what the readback has to show. One
        // `stencil8` texel is one byte, so the expected image is
        // `width * height` bytes — a quarter of the depth sibling's. An
        // expectation that equals the clear image is refused because it could
        // not tell "the pass stored the stencil texels" from "it never wrote
        // stencil" (`docs/23` §3.3, v49).
        let stencilStore: ValidatedStencilStore?
        if let spelling = stencil.store {
            try require(spelling == "store",
                        "\(definition.id): the only stored stencil spelling is \"store\"")
            guard let allocation = stencil.allocation,
                  let view = stencil.view,
                  let expectedHex = stencil.expected_hex else {
                throw OracleError("\(definition.id): a stored stencil attachment needs its "
                                  + "store action, its identity and an expectation")
            }
            try require(allocation > 0 && view > 0,
                        "\(definition.id): zero stencil attachment identity")
            let expected = try decodeHex(expectedHex,
                                         context: "\(definition.id) expected stencil texels")
            try require(expected.count == stencil.width * stencil.height,
                        "\(definition.id): expected stencil texel bytes do not match the "
                        + "stencil attachment")
            // The clear image the surface starts from — one byte per texel, the
            // reviewed clear value — is exactly what a pass that never stored
            // its stencil would read back.
            let clearImage = Data(repeating: UInt8(truncatingIfNeeded: clearValue),
                                  count: stencil.width * stencil.height)
            try require(expected != clearImage,
                        "\(definition.id): the expected stencil texels equal the clear stencil")
            // The stencil landing is a resource of its own: a colour attachment
            // already names these identities, and sharing one would make the
            // colour readback and the stencil readback the same landing
            // (`research/docs/23` §3.3, v49).
            for attachment in attachments {
                try require(allocation != attachment.allocation && view != attachment.view,
                            "\(definition.id): the stencil attachment reuses the colour "
                            + "attachment's identity")
            }
            stencilStore = ValidatedStencilStore(allocation: allocation, view: view,
                                                 expected: expected)
        } else {
            try require(stencil.allocation == nil && stencil.view == nil
                        && stencil.expected_hex == nil,
                        "\(definition.id): a discarded stencil attachment carries no identity "
                        + "or expectation")
            stencilStore = nil
        }
        validatedStencil = ValidatedStencil(width: stencil.width, height: stencil.height,
                                            clearValue: clearValue, reference: test.reference,
                                            store: stencilStore)
    } else {
        validatedStencil = nil
    }
    return ValidatedRender(definition: definition, source: source,
                           attachments: validatedAttachments,
                           vertexStreams: vertexStreams, indexStream: indexStream,
                           depth: validatedDepth, stencil: validatedStencil)
}

private func reviewedProgram(_ entry: String, explicitSlots: Bool = false) throws -> ProgramDefinition {
    let air: SourceDefinition
    let metal: SourceDefinition
    let slots: [BufferSlotDefinition]
    switch entry {
    case "read_texture_2d":
        air = SourceDefinition(path: "../examples/metal-smoke/shaders/kernel_read_texture_2d.ll",
            sha256: "3e969b61d3149bc9351f44c56de6fb85a403557cbcbee7240602539ca794c8df")
        metal = SourceDefinition(path: "shaders/read_texture_2d.metal",
            sha256: "da21ca69d76018f2911aaf6867f517fca8e41b20d531b6b43df30931563499ee")
        slots = [BufferSlotDefinition(binding: 0, access: "write", length: 64)]
    case "read_texture_2d_cell":
        air = SourceDefinition(path: "../examples/metal-smoke/shaders/kernel_read_texture_2d_cell.ll",
            sha256: "80fe6866bac049de9c1c2b33d9f15a3a133b68c321dfdb16721c991f8dfc23c9")
        metal = SourceDefinition(path: "shaders/read_texture_2d_cell.metal",
            sha256: "6517da4354381bb46706ec3395d3e449ff08499df37c0c1f2a620a0c04161237")
        slots = [BufferSlotDefinition(binding: 0, access: "write", length: 64)]
    case "copy_word":
        air = SourceDefinition(path: "../examples/metal-smoke/shaders/kernel_copy_word.ll",
            sha256: "292c3e1ff300fd08bf5e39aaa9abe352842eced807138f863e05056f39c56d99")
        metal = SourceDefinition(path: "shaders/copy_word.metal",
            sha256: "7bfa419aef6eb0abcbec045c1bc15651b2d8f0a7591e07448edc6de6522141bc")
        slots = [BufferSlotDefinition(binding: 0, access: "read", length: 4),
                 BufferSlotDefinition(binding: 1, access: "write", length: 4)]
    case "copy_word_with_witness":
        // v43's declaring kernel reads the whole word of the colour view and of
        // the depth view, which is why both declaring slots are the reviewed
        // 64-byte attachment views (`research/docs/23` §3.3, v43).
        air = SourceDefinition(path: "../examples/metal-smoke/shaders/kernel_copy_word_with_witness.ll",
            sha256: "f24e33124da1c228bf4766d32496d8d7ede4fc29d8e6c582c889a36343dfc18e")
        metal = SourceDefinition(path: "shaders/copy_word_with_witness.metal",
            sha256: "c116fec300f1369069fbcf19d5fbb95e8c5ad07475757c19930075a19ad4367a")
        slots = [BufferSlotDefinition(binding: 0, access: "read", length: 64),
                 BufferSlotDefinition(binding: 1, access: "write", length: 4),
                 BufferSlotDefinition(binding: 2, access: "read", length: 64)]
    case "copy_word_with_witnesses":
        // v60's declaring kernel reads the whole word of the colour view, the
        // depth landing view and the stencil landing view, which is why the
        // three read slots are the three attachment views the combined render
        // pass stores into (`research/docs/23` §3.3, v60).
        air = SourceDefinition(path: "../examples/metal-smoke/shaders/kernel_copy_word_with_witnesses.ll",
            sha256: "a06dcfcf052e51a8b30e42942bee50d620f53e5cca3107bb6779e0d42f43c37e")
        metal = SourceDefinition(path: "shaders/copy_word_with_witnesses.metal",
            sha256: "256da53df3f30d52f0b864d545d015bdaa7e8e45c055f4eb9e1f8a1c3963a335")
        slots = [BufferSlotDefinition(binding: 0, access: "read", length: 64),
                 BufferSlotDefinition(binding: 1, access: "write", length: 4),
                 BufferSlotDefinition(binding: 2, access: "read", length: 64),
                 BufferSlotDefinition(binding: 3, access: "read", length: 16)]
    case "kernel_dispatch_threads_boundary_barrier":
        air = SourceDefinition(path: "../examples/metal-smoke/shaders/kernel_dispatch_threads_boundary_barrier.ll",
            sha256: "95076cf4199734f848fd6d761dce13addc7b55354b4d8ee2be16e59287ea5945")
        metal = SourceDefinition(path: "shaders/indexed_boundary.metal",
            sha256: "7684e493a8704127e39dace5476a006fac564224909c667a57fb5ac9d8291b06")
        slots = [BufferSlotDefinition(binding: 0, access: "write", length: 120)]
    case "transform_3d", "mix_3d":
        if entry == "transform_3d" {
            air = SourceDefinition(path: "shaders/transform_3d.ll",
                sha256: "32bb9a29fef9825972b61cb982106b2bcb7c582413e50350eabc7834532b4df2")
            metal = SourceDefinition(path: "shaders/transform_3d.metal",
                sha256: "5637cf50a3de44568ff7d3b09341e84111e2a9f6ff9b617181c6368efeacaf9b")
        } else {
            air = SourceDefinition(path: "shaders/mix_3d.ll",
                sha256: "cccc601c6f14d5c76808f927118d77cdcb9e4824591c0492faf735197afaf95f")
            metal = SourceDefinition(path: "shaders/mix_3d.metal",
                sha256: "e3fa76b0027e6d20e4649fb6e7c07c0ca1618a9ae88fa13815337d2aa7c99bf5")
        }
        slots = [BufferSlotDefinition(binding: 0, access: "read_write", length: 120),
                 BufferSlotDefinition(binding: 2, access: "read", length: 4),
                 BufferSlotDefinition(binding: 5, access: "write", length: 120)]
    case "remap_3d":
        air = SourceDefinition(path: "shaders/remap_3d.ll",
            sha256: "5388b13783b13a616a3b6952e0c939a120e5d1961e060dd15c11cb54083092ec")
        metal = SourceDefinition(path: "shaders/remap_3d.metal",
            sha256: "0d715fe43e72fd96218f3fefc9a582c8634092fa10cc79a544869b5dee025a76")
        slots = [BufferSlotDefinition(binding: 1, access: "read", length: 4),
                 BufferSlotDefinition(binding: 3, access: "read", length: 120),
                 BufferSlotDefinition(binding: 7, access: "write", length: 120)]
    case "copy_3d":
        air = SourceDefinition(path: "shaders/copy_3d.ll",
            sha256: "9f379575b8f9ed45e62df27c24761d0030e257f45c6241c649b5caae73cbe9cb")
        metal = SourceDefinition(path: "shaders/copy_3d.metal",
            sha256: "3d8d71178abe03067508183a87f8c5c6843f1a3092e7f1cb52471ecaaaf0593f")
        slots = [BufferSlotDefinition(binding: 4, access: "read", length: 120),
                 BufferSlotDefinition(binding: 9, access: "write", length: 120)]
    case "mrt_declare":
        air = SourceDefinition(path: "shaders/mrt_declare.ll",
            sha256: "0a5b6740a2839cc4c47a829a7c9badb1bb7d6557df031620a1bf17d7a04393c9")
        metal = SourceDefinition(path: "shaders/mrt_declare.metal",
            sha256: "c6eeddad6686351c7ec616267f0975f7cc559ee85569a3396c83f64407eff689")
        slots = [BufferSlotDefinition(binding: 0, access: "read", length: 16),
                 BufferSlotDefinition(binding: 1, access: "read", length: 16),
                 BufferSlotDefinition(binding: 2, access: "write", length: 4)]
    case "mrt_declare4":
        air = SourceDefinition(path: "shaders/mrt_declare4.ll",
            sha256: "5bf093fb4ad3890e7ee513591e6b943755db1c6850f091580b658e52a092ccd9")
        metal = SourceDefinition(path: "shaders/mrt_declare4.metal",
            sha256: "b1c51bdf4817b21c9e476eabc627ecb83e727384e3bf2436ac62377702f50a41")
        slots = [BufferSlotDefinition(binding: 0, access: "read", length: 16),
                 BufferSlotDefinition(binding: 1, access: "read", length: 16),
                 BufferSlotDefinition(binding: 2, access: "read", length: 16),
                 BufferSlotDefinition(binding: 3, access: "read", length: 16),
                 BufferSlotDefinition(binding: 4, access: "write", length: 4)]
    default:
        throw OracleError("Unsupported entry: \(entry)")
    }
    return ProgramDefinition(entry: entry, air: air, metal: metal, buffer_slots: explicitSlots ? slots : nil)
}

private func validatePrograms(_ definition: CaseDefinition, suite: String) throws -> [ProgramDefinition] {
    let primary = ProgramDefinition(entry: definition.entry, air: definition.air,
                                    metal: definition.metal, buffer_slots: nil)
    if suite == "compute-buffer-v7" || suite == "compute-buffer-v8" || suite == "compute-buffer-v9" {
        guard let supplied = definition.programs else { throw OracleError("Program table required") }
        var expected = try [reviewedProgram("transform_3d", explicitSlots: true),
                            reviewedProgram("copy_3d", explicitSlots: true)]
        if definition.id != "subset_chain_two" {
            expected.append(try reviewedProgram("remap_3d", explicitSlots: true))
        }
        try require(primary == (try reviewedProgram("transform_3d")) && supplied == expected,
                    "Unreviewed subset program table or buffer layout")
        return supplied
    }
    if suite == "compute-buffer-v5" || suite == "compute-buffer-v6" {
        guard let supplied = definition.programs else { throw OracleError("Program table required") }
        let layouts = suite == "compute-buffer-v6"
        let expected = try [reviewedProgram("transform_3d", explicitSlots: layouts),
                            reviewedProgram(layouts ? "remap_3d" : "mix_3d", explicitSlots: layouts)]
        try require(primary == (try reviewedProgram("transform_3d")) && supplied == expected,
                    "Unreviewed program table or buffer layout")
        return supplied
    }
    try require(definition.programs == nil, "Legacy case cannot carry program table")
    try require(primary == (try reviewedProgram(definition.entry)), "Unreviewed primary program")
    return [primary]
}

@available(macOS 11.0, *)
private func loadProgram(_ program: ProgramDefinition, root: URL) throws -> String {
    let reviewed = try reviewedProgram(program.entry)
    _ = try validateSource(program.air, root: root, path: reviewed.air.path, digest: reviewed.air.sha256)
    let metalBytes = try validateSource(program.metal, root: root, path: reviewed.metal.path, digest: reviewed.metal.sha256)
    guard let metalSource = String(data: metalBytes, encoding: .utf8) else {
        throw OracleError("\(program.entry): MSL source is not UTF-8")
    }
    return metalSource
}

private func metalSize(_ dimensions: [UInt64]) -> MTLSize {
    MTLSize(width: Int(dimensions[0]), height: Int(dimensions[1]), depth: Int(dimensions[2]))
}

@available(macOS 11.0, *)
private func runCase(_ fixture: ValidatedCase, device: MTLDevice, queue: MTLCommandQueue,
                     pipelines: [MTLComputePipelineState]) throws -> CaseResult {
    let definition = fixture.definition
    try require(!fixture.dispatches.isEmpty && fixture.dispatches.count <= maximumPassCount,
                "\(definition.id): runtime supports one to eight serial passes")
    let limits = device.maxThreadsPerThreadgroup
    // Check every dispatch before allocating buffers or creating commands.
    for (index, dispatch) in fixture.dispatches.enumerated() {
        let local = metalSize(dispatch.local)
        try require(local.width <= limits.width && local.height <= limits.height && local.depth <= limits.depth,
                    "\(definition.id) pass \(index): local dimensions exceed device limits")
        try require(local.width * local.height * local.depth <= pipelines[dispatch.program ?? 0].maxTotalThreadsPerThreadgroup,
                    "\(definition.id) pass \(index): local thread count exceeds pipeline limit")
    }

    var resources = [MTLBuffer]()
    for buffer in fixture.buffers {
        try require(buffer.backing.count <= device.maxBufferLength,
                    "\(definition.id): allocation exceeds the Metal buffer limit")
        guard let resource = device.makeBuffer(length: buffer.backing.count, options: .storageModeShared) else {
            throw OracleError("\(definition.id): cannot allocate a shared Metal buffer")
        }
        try require(resource.storageMode == .shared, "\(definition.id): shared storage was not selected")
        try require(resource.hazardTrackingMode == .tracked,
                    "\(definition.id): automatic resource hazard tracking is required")
        buffer.backing.withUnsafeBytes { bytes in
            if let source = bytes.baseAddress {
                resource.contents().copyMemory(from: source, byteCount: bytes.count)
            }
        }
        resources.append(resource)
    }
    // v11: sampled textures are their own allocations. The AIR fixture reads
    // one R32Uint texel per thread, so the MTLTexture is filled once from the
    // case's tightly packed initial bytes.
    var textures = [MTLTexture]()
    for texture in fixture.textures {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Uint,
            width: texture.definition.width,
            height: texture.definition.height,
            mipmapped: false)
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared
        guard let resource = device.makeTexture(descriptor: descriptor) else {
            throw OracleError("\(definition.id): cannot allocate the sampled texture")
        }
        texture.backing.withUnsafeBytes { bytes in
            if let source = bytes.baseAddress {
                resource.replace(region: MTLRegionMake2D(0, 0, texture.definition.width,
                                                         texture.definition.height),
                                 mipmapLevel: 0,
                                 withBytes: source,
                                 bytesPerRow: texture.definition.width * 4)
            }
        }
        textures.append(resource)
    }
    // makeCommandBuffer() retains referenced resources until GPU completion.
    // In particular, a CPU timeout below must not release submitted buffers.
    // One command buffer commits and completes before the next group is
    // recorded, so later commands observe earlier landed writes.
    for (groupIndex, group) in fixture.commandBuffers.enumerated() {
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw OracleError("\(definition.id): cannot create a command buffer")
        }
        try require(commandBuffer.retainedReferences, "\(definition.id): command buffer does not retain resources")
        commandBuffer.label = "native oracle: \(definition.id) cb\(groupIndex)"
        // The default encoder is serial. Direct bindings of device-created tracked
        // buffers let MTLCommandQueue synchronize writes between successive passes:
        // https://developer.apple.com/documentation/metal/resource-synchronization
        // Keep one buffer set across command buffers so no CPU upload resets earlier writes.
        for dispatchIndex in group {
            let dispatch = fixture.dispatches[dispatchIndex]
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw OracleError("\(definition.id): cannot create a compute encoder")
            }
            encoder.setComputePipelineState(pipelines[dispatch.program ?? 0])
            for (index, slot) in bufferSlots(definition, program: dispatch.program ?? 0).enumerated() {
                let view = dispatch.bindings?[index] ?? fixture.buffers[index].definition.view
                guard let poolIndex = fixture.buffers.firstIndex(where: { $0.definition.view == view }) else {
                    throw OracleError("Unknown bound resource")
                }
                let resource = fixture.buffers[poolIndex]
                encoder.setBuffer(resources[poolIndex], offset: Int(resource.definition.offset), index: Int(slot.binding))
            }
            for (index, texture) in fixture.textures.enumerated() {
                encoder.setTexture(textures[index], index: texture.definition.binding)
            }
            encoder.dispatchThreads(metalSize(dispatch.grid), threadsPerThreadgroup: metalSize(dispatch.local))
            encoder.endEncoding()
        }

        let completed = DispatchSemaphore(value: 0)
        commandBuffer.addCompletedHandler { _ in completed.signal() }
        commandBuffer.commit()
        guard completed.wait(timeout: .now() + .seconds(20)) == .success else {
            // Throwing reaches the top-level nonzero exit. No other case is run,
            // buffers are not inspected, and no partial report is published.
            throw OracleError("\(definition.id): GPU completion timed out after 20 seconds; submitted work was not cancelled")
        }
        try require(commandBuffer.status == .completed && commandBuffer.error == nil,
                    "\(definition.id): Metal execution failed (status \(commandBuffer.status.rawValue)): \(String(describing: commandBuffer.error))")
    }

    var writebacks = [Writeback]()
    let writtenViews = writableViews(definition)
    // Several views may share one allocation. Each view owns its own Metal
    // buffer over the shared image, so the reported allocation is composed by
    // overlaying every view's observed bytes onto that one extent.
    var observedImages = [UInt64: Data]()
    for (buffer, resource) in zip(fixture.buffers, resources) {
        let specification = buffer.definition
        // Only completed shared resources are CPU-visible. Copy the complete
        // backing allocation so the comparator can independently inspect it.
        let observed = Data(bytes: resource.contents(), count: buffer.backing.count)
        let start = Int(specification.offset)
        let end = start + Int(specification.length)
        if !writtenViews.contains(specification.view) {
            try require(observed == buffer.backing, "\(definition.id): read-only allocation \(specification.allocation) changed")
        } else {
            try require(observed.prefix(start) == buffer.backing.prefix(start)
                        && observed.suffix(observed.count - end) == buffer.backing.suffix(buffer.backing.count - end),
                        "\(definition.id): guard bytes changed in allocation \(specification.allocation)")
            writebacks.append(Writeback(allocation: specification.allocation, view: specification.view,
                offset: specification.offset, bytes_hex: hex(observed.subdata(in: start..<end))))
        }
        var image = observedImages[specification.allocation] ?? buffer.backing
        image.replaceSubrange(start..<end, with: observed.subdata(in: start..<end))
        observedImages[specification.allocation] = image
    }
    var allocations = [AllocationResult]()
    for allocation in observedImages.keys.sorted() {
        allocations.append(AllocationResult(allocation: allocation, bytes_hex: hex(observedImages[allocation]!)))
    }
    writebacks.sort { ($0.allocation, $0.view) < ($1.allocation, $1.view) }
    return CaseResult(id: definition.id, completion: "CompletedVisible", writebacks: writebacks, allocations: allocations)
}

/// The resolve of one texel's samples (`research/docs/23` §3.3, v51).
///
/// `covered` of `samples` samples carry the fragment output and the rest the
/// colour the pass started from, so each resolved channel is their arithmetic
/// mean. A mean that is not exactly representable answers `nil` rather than a
/// rounded byte — the same rule `conformance/compare.py` states, and what keeps
/// the expectation independent of a driver's rounding rule.
private func resolveTexel(fragment: Data, clear: Data, covered: Int, samples: Int) -> Data? {
    var resolved = Data(capacity: 4)
    for channel in 0..<4 {
        let total = Int(fragment[channel]) * covered + Int(clear[channel]) * (samples - covered)
        if total % samples != 0 {
            return nil
        }
        resolved.append(UInt8(total / samples))
    }
    return resolved
}

private func hostOffset(_ value: UInt64, id: String) throws -> Int {
    // A view offset is a wire `u64`, so the narrowing is fallible by
    // construction; refusing it is the same answer `render.rs::stream_buffer`
    // gives for the same field instead of trapping on the conversion.
    guard let narrowed = Int(exactly: value) else {
        throw OracleError("\(id): a view offset does not fit this host")
    }
    return narrowed
}

/// One `MTLBuffer` holding a stream view's bytes at the view's own offset.
///
/// The image is the view's bytes placed at `offset` inside its allocation, and
/// the binding points at that same offset — the convention the provider rail's
/// `render.rs::stream_buffer` follows, so a view that starts above the
/// allocation's first byte still reads the byte range the case named instead of
/// being silently re-based at zero.
@available(macOS 11.0, *)
private func makeStreamBuffer(device: MTLDevice, id: String,
                              offset: UInt64, bytes: Data) throws -> MTLBuffer {
    guard !bytes.isEmpty else {
        throw OracleError("\(id): a stream view carries no bytes")
    }
    let start = try hostOffset(offset, id: id)
    guard start <= Int.max - bytes.count else {
        throw OracleError("\(id): a stream view offset does not fit this host")
    }
    var image = Data(count: start + bytes.count)
    image.replaceSubrange(start..<(start + bytes.count), with: bytes)
    let buffer: MTLBuffer? = image.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
        guard let base = raw.baseAddress else { return nil }
        return device.makeBuffer(bytes: base, length: image.count, options: .storageModeShared)
    }
    guard let allocated = buffer else {
        throw OracleError("\(id): cannot allocate a stream buffer")
    }
    return allocated
}

/// One offscreen render case: a 2x2 `rgba8Unorm` attachment, the reviewed
/// two-entry pipeline and either the full-screen-triangle draw or the indexed
/// quad, read back as texels.
///
/// The observable is the attachment's tightly packed texels, reported in the
/// same `writebacks`/`allocations` shape every other case uses, so a render case
/// needs no second observation channel (`research/docs/23` §1.1). The milestone
/// assertion is falsifiable in two ways: the expectation covers every texel, and
/// it has to differ from the sentinel the pass started from.
@available(macOS 11.0, *)
private func runRenderCase(_ fixture: ValidatedRender, device: MTLDevice,
                           queue: MTLCommandQueue) throws -> CaseResult {
    let definition = fixture.definition
    // Every pre-v31 case draws a single instance; the reviewed instanced case
    // declares two (`research/docs/23` §3.3, v31).
    let instanceCount = Int(definition.instance_count ?? 1)
    // One texture per colour attachment, in location order. The attachments are
    // render targets, not sampled sources. Shared storage is what makes their
    // texels CPU-visible for the readback on the unified-memory device this
    // oracle requires, the same reason the sampled texture rail uses it
    // (`research/docs/16` §4.8).
    var targets = [MTLTexture]()
    for attachment in fixture.attachments {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: attachment.pixelFormat,
            width: attachment.width,
            height: attachment.height,
            mipmapped: false)
        descriptor.usage = .renderTarget
        descriptor.storageMode = .shared
        guard let target = device.makeTexture(descriptor: descriptor) else {
            throw OracleError("\(definition.id): cannot allocate the colour attachment")
        }
        target.label = "native oracle: \(definition.id)"
        if let initial = attachment.initial {
            initial.withUnsafeBytes { bytes in
                if let source = bytes.baseAddress {
                    target.replace(region: MTLRegionMake2D(0, 0, attachment.width, attachment.height),
                                   mipmapLevel: 0,
                                   withBytes: source,
                                   bytesPerRow: attachment.width * 4)
                }
            }
        }
        targets.append(target)
    }
    // A multisampled pass renders into its own four-sample textures
    // (`research/docs/23` §3.3, v51), one per colour location; `targets` above
    // are the resolve targets the pass writes into and the readback below
    // observes. The four-sample surfaces are never read back, so private
    // storage is enough, exactly as the discarded depth surface above states
    // it. The local keeps them alive until the encoder's own reference takes
    // over.
    var multisampleTargets = [MTLTexture]()
    if let multisample = definition.multisample {
        let samples = Int(multisample.sample_count)
        for attachment in fixture.attachments {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: attachment.pixelFormat,
                width: attachment.width,
                height: attachment.height,
                mipmapped: false)
            descriptor.textureType = .type2DMultisample
            descriptor.sampleCount = samples
            descriptor.usage = .renderTarget
            descriptor.storageMode = .private
            guard let target = device.makeTexture(descriptor: descriptor) else {
                throw OracleError("\(definition.id): cannot allocate the multisample attachment")
            }
            target.label = "native oracle: \(definition.id) msaa"
            multisampleTargets.append(target)
        }
    }
    // The depth surface is the rail's own texture for the shape every pre-v43
    // case declares (`research/docs/23` §3.3, v36): no trace identity and no
    // readback, so private storage is enough and the pass discards it after
    // the draw, exactly as `render.rs::depth_texture` creates it. A case that
    // stores the surface (`§3.3`, v43) reads its texels back on the CPU, and
    // `getBytes` cannot read a private texture (Apple's `MTLTexture`
    // documentation): that shape allocates shared storage, the same reason its
    // colour attachments do. The local keeps the texture alive until the
    // encoder's own reference takes over.
    var depthTarget: MTLTexture?
    if let depth = fixture.depth {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            // The combined shape's two faces share one `depth32Float_stencil8`
            // texture; the single-face depth shape keeps `depth32Float`
            // (`research/docs/23` §3.3, v60).
            pixelFormat: fixture.stencil != nil ? .depth32Float_stencil8 : .depth32Float,
            width: depth.width,
            height: depth.height,
            mipmapped: false)
        // A multisampled pass creates its depth surface with the raster's own
        // sample count (`research/docs/23` §3.3, v53): Metal refuses an encoder
        // whose depth texture disagrees with `rasterSampleCount`. A resolving
        // pass (v57c) keeps that surface private — the resolve target below is
        // what the CPU reads back — while a stored non-resolving surface keeps
        // its own shared texels.
        if let multisample = definition.multisample {
            descriptor.textureType = .type2DMultisample
            descriptor.sampleCount = Int(multisample.sample_count)
        }
        descriptor.usage = .renderTarget
        descriptor.storageMode = (depth.store == nil || definition.depth_resolve != nil)
            ? .private : .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw OracleError("\(definition.id): cannot allocate the depth attachment")
        }
        texture.label = "native oracle: \(definition.id) depth"
        depthTarget = texture
    }
    // The single-sample landing a depth resolve writes into
    // (`research/docs/23` §3.3, v57c): the v43 shared-storage readback texture,
    // one `depth32Float` texel per texel, observed by the readback below
    // exactly as a stored non-resolving surface observes its own texture. The
    // four-sample surface above never reaches the CPU, so it stays private.
    var depthResolveTarget: MTLTexture?
    if let depth = fixture.depth, definition.depth_resolve != nil {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float,
            width: depth.width,
            height: depth.height,
            mipmapped: false)
        descriptor.usage = .renderTarget
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw OracleError("\(definition.id): cannot allocate the depth resolve target")
        }
        texture.label = "native oracle: \(definition.id) depth resolve"
        depthResolveTarget = texture
    }
    // The stencil surface is the rail's own texture for the shape every pre-v49
    // case declares (`research/docs/23` §3.3, v47): the stored values decide
    // which primitives survive and nothing reads them back, so private storage
    // is enough and the pass discards it after the draw — the same shape the
    // pre-v43 depth surface has. A case that stores the surface (`§3.3`, v49)
    // reads its texels back on the CPU, and `getBytes` cannot read a private
    // texture (Apple's `MTLTexture` documentation): that shape allocates shared
    // storage, the same reason its colour attachments and the storing depth
    // surface do. The local keeps the texture alive until the encoder's own
    // reference takes over.
    var stencilTarget: MTLTexture?
    if let stencil = fixture.stencil {
        // The combined shape reuses the depth texture the branch above built:
        // Metal binds one texture to both attachment descriptors.
        if fixture.depth != nil {
            stencilTarget = depthTarget
            guard stencilTarget != nil else {
                throw OracleError("\(definition.id): cannot share the combined surface")
            }
        } else {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .stencil8,
            width: stencil.width,
            height: stencil.height,
            mipmapped: false)
        // A multisampled pass creates its stencil surface with the raster's own
        // sample count (`research/docs/23` §3.3, v55), exactly as the depth
        // surface does; the surface is rail-owned, so private storage is enough.
        if let multisample = definition.multisample {
            descriptor.textureType = .type2DMultisample
            descriptor.sampleCount = Int(multisample.sample_count)
        }
        descriptor.usage = .renderTarget
        descriptor.storageMode = stencil.store == nil ? .private : .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw OracleError("\(definition.id): cannot allocate the stencil attachment")
        }
        texture.label = "native oracle: \(definition.id) stencil"
        stencilTarget = texture
        }
    }
    // The single-sample landing a stencil resolve writes into
    // (`research/docs/23` §3.3, v60): the v49 shared-storage readback texture,
    // one `stencil8` byte per texel, observed by the readback below exactly as
    // a stored non-resolving surface observes its own texture.
    var stencilResolveTarget: MTLTexture?
    if let stencil = fixture.stencil, definition.stencil_resolve != nil {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .stencil8,
            width: stencil.width,
            height: stencil.height,
            mipmapped: false)
        descriptor.usage = .renderTarget
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw OracleError("\(definition.id): cannot allocate the stencil resolve target")
        }
        texture.label = "native oracle: \(definition.id) stencil resolve"
        stencilResolveTarget = texture
    }
    // The two stage entries come from the one reviewed module; `loadSuite`
    // already proved the identity, so only the lookup can still fail.
    let library = try device.makeLibrary(source: fixture.source, options: nil)
    guard let vertexFunction = library.makeFunction(name: definition.vertex_entry) else {
        throw OracleError("\(definition.id): vertex entry \(definition.vertex_entry) was not found")
    }
    guard let fragmentFunction = library.makeFunction(name: definition.fragment_entry) else {
        throw OracleError("\(definition.id): fragment entry \(definition.fragment_entry) was not found")
    }
    let pipelineDescriptor = MTLRenderPipelineDescriptor()
    pipelineDescriptor.label = "native oracle: \(definition.id)"
    pipelineDescriptor.vertexFunction = vertexFunction
    pipelineDescriptor.fragmentFunction = fragmentFunction
    // The vertex descriptor is what makes `[[attribute(n)]]` mean a byte range
    // of a bound stream: the reviewed module names the attribute locations, this
    // descriptor says which binding, stride, offset and format each one reads.
    // The `vertex_id` fixture binds no stream and carries none, the same split
    // `render.rs::render_pipeline_state` makes.
    var streamBuffers = [MTLBuffer]()
    if !fixture.vertexStreams.isEmpty {
        let vertexDescriptor = MTLVertexDescriptor()
        for stream in fixture.vertexStreams {
            // The descriptor's subscript is an implicitly unwrapped optional on
            // the Swift side of Metal; referencing a member before unwrapping is
            // a compile error under `-warnings-as-errors`, so unwrap explicitly.
            guard let layout = vertexDescriptor.layouts[stream.binding] else {
                throw OracleError("\(definition.id): the vertex descriptor has no layout "
                                  + "\(stream.binding)")
            }
            layout.stride = Int(stream.stride)
            // The v31 stream advance (`research/docs/23` §3.3): a per-instance
            // stream advances once per instance, every other stream once per
            // vertex. Rate one is the reviewed step, not a knob: no reviewed
            // fixture declares a wider rate.
            if stream.step == "per_instance" {
                layout.stepFunction = .perInstance
                layout.stepRate = 1
            } else {
                layout.stepFunction = .perVertex
            }
            for attribute in stream.attributes {
                guard let format = vertexFormat(attribute.format) else {
                    throw OracleError("\(definition.id): unsupported vertex attribute format "
                                      + attribute.format)
                }
                guard let target = vertexDescriptor.attributes[Int(attribute.location)] else {
                    throw OracleError("\(definition.id): the vertex descriptor has no attribute "
                                      + "\(attribute.location)")
                }
                target.format = format
                target.offset = Int(attribute.offset)
                target.bufferIndex = stream.binding
            }
            streamBuffers.append(try makeStreamBuffer(device: device, id: definition.id,
                                                      offset: stream.offset,
                                                      bytes: stream.bytes))
        }
        pipelineDescriptor.vertexDescriptor = vertexDescriptor
    }
    // One pipeline attachment per colour location: entry `i` states the pixel
    // format the reviewed fragment's output `i` is compiled against, which the
    // validation above already forced to agree with the case's attachment list.
    // The zero-colour shape (`research/docs/23` §3.3, v46) states no location at
    // all: its fragment entry generates no output, so there is no colour format
    // to compile against and this loop runs zero times.
    for index in 0..<fixture.attachments.count {
        pipelineDescriptor.colorAttachments[index].pixelFormat =
            fixture.attachments[index].pixelFormat
    }
    // A pass that states blend state states it here, on its pipeline's colour
    // attachment (`research/docs/23` §3.3, v40): Metal keeps the factors and
    // the operation in the pipeline descriptor rather than on the encoder,
    // exactly as `render.rs::metal_blend_factor` / `metal_blend_operation`
    // state them on `MTLRenderPipelineColorAttachmentDescriptor`. The
    // validation above admitted exactly one state — one colour attachment
    // blending source alpha against one-minus-source-alpha with an add — so
    // the reviewed state below is that shape by construction, and every pre-v40
    // case leaves blending at Metal's own default (off).
    if definition.blend != nil {
        guard let color = pipelineDescriptor.colorAttachments[0] else {
            throw OracleError("\(definition.id): cannot reach the blended colour attachment")
        }
        color.isBlendingEnabled = true
        color.rgbBlendOperation = .add
        color.alphaBlendOperation = .add
        color.sourceRGBBlendFactor = .sourceAlpha
        color.destinationRGBBlendFactor = .oneMinusSourceAlpha
        color.sourceAlphaBlendFactor = .sourceAlpha
        color.destinationAlphaBlendFactor = .oneMinusSourceAlpha
    }
    // A pass that opens a depth attachment compiles its pipeline against that
    // attachment's format; every pre-v36 case declares none
    // (`research/docs/23` §3.3, v36).
    if fixture.depth != nil {
        pipelineDescriptor.depthAttachmentPixelFormat =
            fixture.stencil != nil ? .depth32Float_stencil8 : .depth32Float
    }
    // The stencil sibling (`research/docs/23` §3.3, v47): a pass that opens a
    // stencil attachment compiles its pipeline against that attachment's
    // format, exactly as the depth branch above states the depth format.
    if fixture.stencil != nil {
        pipelineDescriptor.stencilAttachmentPixelFormat =
            fixture.depth != nil ? .depth32Float_stencil8 : .stencil8
    }
    // The pipeline's raster sample count follows the pass's own multisample
    // state (`research/docs/23` §3.3, v51): Metal refuses a pipeline whose
    // `rasterSampleCount` disagrees with the attachments the pass binds, so the
    // two come from one decision. A single-sample case keeps the default
    // exactly as every pre-v51 case did.
    if let multisample = definition.multisample {
        pipelineDescriptor.rasterSampleCount = Int(multisample.sample_count)
    }
    let pipeline = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)

    let pass = MTLRenderPassDescriptor()
    // `colorAttachments[i]` is an implicitly unwrapped optional on the Swift
    // side of Metal, but referencing a member before unwrapping is a compile
    // error under `-warnings-as-errors`; unwrap each entry explicitly.
    for (index, attachment) in fixture.attachments.enumerated() {
        guard let color = pass.colorAttachments[index] else {
            throw OracleError("\(definition.id): cannot reach colour attachment \(index)")
        }
        // A multisampled location names the four-sample surface as its render
        // target and the attachment's own texture as the resolve target
        // (`research/docs/23` §3.3, v51): `.multisampleResolve` writes the
        // resolved texels into the latter, which is what the readback below
        // observes. A single-sample location names one texture, exactly as
        // every pre-v51 case did.
        if definition.multisample != nil {
            color.texture = multisampleTargets[index]
            color.resolveTexture = targets[index]
        } else {
            color.texture = targets[index]
        }
        // A discarded attachment still renders, but Metal does not keep its
        // bytes: `.dontCare` is what makes it disappear from the observable
        // surface, and the readback below skips it (`research/docs/23` §3.6,
        // v19).
        if definition.multisample != nil {
            color.storeAction = attachment.store == "store" ? .multisampleResolve : .dontCare
        } else {
            color.storeAction = attachment.store == "store" ? .store : .dontCare
        }
        switch attachment.load {
        case "clear":
            color.loadAction = .clear
            guard attachment.clearComponents.count == 4 else {
                throw OracleError("\(definition.id): a clear colour is four components")
            }
            color.clearColor = MTLClearColor(red: attachment.clearComponents[0],
                                             green: attachment.clearComponents[1],
                                             blue: attachment.clearComponents[2],
                                             alpha: attachment.clearComponents[3])
        case "load":
            color.loadAction = .load
        case "dontcare":
            // Undefined pre-pass contents (`docs/23` §13, v20): the pass opens
            // the attachment from nothing, exactly as `LoadOp::DontCare` does
            // on the provider rails, and the shared texture above was created
            // without any pre-seed.
            color.loadAction = .dontCare
        default:
            throw OracleError("\(definition.id): unsupported attachment load op \(attachment.load)")
        }
    }
    // The pass opens the depth surface with its own load operation
    // (`research/docs/23` §3.3, v36) and keeps it only when the case says so
    // (`§3.3`, v43): a stored surface lands in the identity its case states,
    // and the rail-owned shape every pre-v43 case declares discards its texels
    // with the pass — `dontCare` is that store action, exactly as `render.rs`
    // opens the plan's depth attachment.
    if let depth = fixture.depth {
        guard let texture = depthTarget, let attachment = pass.depthAttachment else {
            throw OracleError("\(definition.id): cannot reach the depth attachment")
        }
        attachment.texture = texture
        attachment.loadAction = .clear
        attachment.clearDepth = depth.clearDepth
        // A resolving stored surface lands through `.multisampleResolve` in the
        // single-sample target above, with the filter the case named
        // (`research/docs/23` §3.3, v57c); every other stored shape keeps its
        // own texels, and a discarded surface disappears with the pass
        // (`§3.3`, v43).
        if let resolve = definition.depth_resolve, let landing = depthResolveTarget {
            attachment.storeAction = .multisampleResolve
            attachment.resolveTexture = landing
            switch resolve.filter {
            case "sample0":
                attachment.depthResolveFilter = .sample0
            case "min":
                attachment.depthResolveFilter = .min
            case "max":
                attachment.depthResolveFilter = .max
            default:
                throw OracleError(
                    "\(definition.id): unsupported depth resolve filter \(resolve.filter)")
            }
        } else {
            attachment.storeAction = depth.store == nil ? .dontCare : .store
        }
    }
    // The stencil surface opens with its own load operation (`research/docs/23`
    // §3.3, v47) and leaves with the pass unless the case says otherwise
    // (`§3.3`, v49): a stored surface lands in the identity its case states,
    // and the rail-owned shape every pre-v49 case declares discards its texels
    // with the pass — `dontCare` is that store action, the same one the
    // rail-owned depth surface states above.
    if let stencil = fixture.stencil {
        guard let texture = stencilTarget, let attachment = pass.stencilAttachment else {
            throw OracleError("\(definition.id): cannot reach the stencil attachment")
        }
        attachment.texture = texture
        attachment.loadAction = .clear
        attachment.clearStencil = stencil.clearValue
        // A resolving stored surface lands through `.multisampleResolve` in
        // the single-sample target above, with the filter the case named
        // (`research/docs/23` §3.3, v60); every other stored shape keeps its
        // own texels, and a discarded surface disappears with the pass.
        if let resolve = definition.stencil_resolve, let landing = stencilResolveTarget {
            attachment.storeAction = .multisampleResolve
            attachment.resolveTexture = landing
            switch resolve.filter {
            case "sample0":
                attachment.stencilResolveFilter = .sample0
            case "depth_resolved_sample":
                attachment.stencilResolveFilter = .depthResolvedSample
            default:
                throw OracleError(
                    "\(definition.id): unsupported stencil resolve filter \(resolve.filter)")
            }
        } else {
            attachment.storeAction = stencil.store == nil ? .dontCare : .store
        }
    }
    guard let commandBuffer = queue.makeCommandBuffer() else {
        throw OracleError("\(definition.id): cannot create a command buffer")
    }
    try require(commandBuffer.retainedReferences,
                "\(definition.id): command buffer does not retain resources")
    commandBuffer.label = "native oracle: \(definition.id)"
    guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
        throw OracleError("\(definition.id): cannot create a render encoder")
    }
    encoder.setRenderPipelineState(pipeline)
    // Metal's depth state is encoder state (`research/docs/23` §3.3, v36): the
    // reviewed pair is a `less` compare with writes on, and the descriptor is
    // where the compare function and the write flag are stated.
    if let depth = fixture.depth {
        let depthStencilDescriptor = MTLDepthStencilDescriptor()
        depthStencilDescriptor.depthCompareFunction = depth.isLess ? .less : .always
        depthStencilDescriptor.isDepthWriteEnabled = depth.write
        // The combined shape carries the stencil half in the one depth-stencil
        // descriptor: the reviewed always test that increments on depth pass
        // and keeps on depth fail (`research/docs/23` §3.3, v60).
        if let stencil = fixture.stencil {
            let stencilDescriptor = MTLStencilDescriptor()
            stencilDescriptor.stencilCompareFunction = .always
            stencilDescriptor.stencilFailureOperation = .keep
            stencilDescriptor.depthFailureOperation = .keep
            stencilDescriptor.depthStencilPassOperation = .incrementWrap
            stencilDescriptor.readMask = 255
            stencilDescriptor.writeMask = 255
            depthStencilDescriptor.frontFaceStencil = stencilDescriptor
            depthStencilDescriptor.backFaceStencil = stencilDescriptor
            encoder.setStencilReferenceValue(stencil.reference)
        }
        guard let state = device.makeDepthStencilState(descriptor: depthStencilDescriptor) else {
            throw OracleError("\(definition.id): cannot create the depth-stencil state")
        }
        encoder.setDepthStencilState(state)
    }
    // The stencil pair's state is encoder state too (`research/docs/23` §3.3,
    // v47): the reviewed state is the same `MTLStencilDescriptor` on the front
    // and the back face — an `equal` test with both masks wide open that keeps
    // both failure outcomes and increments-wraps on pass — and the reference
    // value travels with the encoder rather than the descriptor. The depth half
    // of the descriptor keeps Metal's own defaults, exactly as a pass without a
    // depth attachment does.
    if let stencil = fixture.stencil, fixture.depth == nil {
        let stencilDescriptor = MTLStencilDescriptor()
        stencilDescriptor.stencilCompareFunction = .equal
        stencilDescriptor.stencilFailureOperation = .keep
        stencilDescriptor.depthFailureOperation = .keep
        stencilDescriptor.depthStencilPassOperation = .incrementWrap
        stencilDescriptor.readMask = 255
        stencilDescriptor.writeMask = 255
        let depthStencilDescriptor = MTLDepthStencilDescriptor()
        depthStencilDescriptor.frontFaceStencil = stencilDescriptor
        depthStencilDescriptor.backFaceStencil = stencilDescriptor
        guard let state = device.makeDepthStencilState(descriptor: depthStencilDescriptor) else {
            throw OracleError("\(definition.id): cannot create the stencil state")
        }
        encoder.setDepthStencilState(state)
        encoder.setStencilReferenceValue(stencil.reference)
    }
    // Culling is encoder state too (`research/docs/23` §3.3, v39): the mode and
    // the winding are the pass's own, and a pass without the state keeps
    // Metal's defaults (cull none, counter-clockwise front) exactly. The
    // validation above pinned the reviewed pair to a back-face cull with a
    // counter-clockwise front.
    if definition.cull != nil {
        encoder.setCullMode(.back)
        // `setFrontFacingWinding(_:)` was renamed to `setFrontFacing(_:)`; the
        // ObjC selector behind it is the same state the contract names.
        encoder.setFrontFacing(.counterClockwise)
    }
    // The viewport is explicit because the contract carries it, even though the
    // first increment only accepts the attachment-covering default. The render
    // area is the colour attachment's extent when the pass has one, and the
    // depth surface's own for the zero-colour shape (`research/docs/23` §3.3,
    // v46) — the validation above pinned the case's viewport to whichever of
    // the two the pass carries.
    guard let renderWidth = fixture.attachments.first?.width ?? fixture.depth?.width,
          let renderHeight = fixture.attachments.first?.height ?? fixture.depth?.height else {
        throw OracleError("\(definition.id): the pass states no render area")
    }
    encoder.setViewport(MTLViewport(originX: 0, originY: 0,
                                    width: Double(renderWidth),
                                    height: Double(renderHeight),
                                    znear: 0, zfar: 1))
    // The scissor is the pass's own rectangle when it declares one; both rails
    // state it in framebuffer coordinates with the origin at the render area's
    // top-left (`research/docs/23` §3.3, v29).
    if let scissor = definition.scissor {
        encoder.setScissorRect(MTLScissorRect(x: Int(scissor[0]), y: Int(scissor[1]),
                                              width: Int(scissor[2]), height: Int(scissor[3])))
    }
    // The streams are bound at the same indices the descriptor names, and they
    // stay alive until the command buffer has completed (the buffers array is
    // released after the readback below).
    for (stream, buffer) in zip(fixture.vertexStreams, streamBuffers) {
        encoder.setVertexBuffer(buffer,
                                offset: try hostOffset(stream.offset, id: definition.id),
                                index: stream.binding)
    }
    if let indexStream = fixture.indexStream {
        // An indexed draw names its index buffer in the draw call, and the count
        // is the one the case declares for that shape
        // (`RenderPassDescriptor::vertices`, `research/docs/23` §3.3); the
        // instance count is the case's own, one for every pre-v31 case.
        let indexBuffer = try makeStreamBuffer(device: device, id: definition.id,
                                               offset: indexStream.offset,
                                               bytes: indexStream.bytes)
        streamBuffers.append(indexBuffer)
        let indexBufferOffset = try hostOffset(indexStream.offset, id: definition.id)
        if indexStream.baseVertex == 0 {
            encoder.drawIndexedPrimitives(type: .triangle,
                                          indexCount: Int(indexStream.indexCount),
                                          indexType: indexStream.format.metal,
                                          indexBuffer: indexBuffer,
                                          indexBufferOffset: indexBufferOffset,
                                          instanceCount: instanceCount)
        } else {
            // The offset belongs to the draw call (`research/docs/23` §3.3,
            // v34): the base-vertex entry states it beside the instance count,
            // and a zero-offset draw keeps the pre-v34 entry point exactly.
            // The wire field is a `u64`, so the narrowing is fallible by
            // construction; refusing it fails closed instead of trapping.
            guard let baseVertex = Int(exactly: indexStream.baseVertex) else {
                throw OracleError("\(definition.id): the base vertex does not fit this host")
            }
            encoder.drawIndexedPrimitives(type: .triangle,
                                          indexCount: Int(indexStream.indexCount),
                                          indexType: indexStream.format.metal,
                                          indexBuffer: indexBuffer,
                                          indexBufferOffset: indexBufferOffset,
                                          instanceCount: instanceCount,
                                          baseVertex: baseVertex,
                                          baseInstance: 0)
        }
    } else {
        encoder.drawPrimitives(type: .triangle, vertexStart: 0,
                               vertexCount: Int(definition.vertices),
                               instanceCount: instanceCount)
    }
    encoder.endEncoding()
    let completed = DispatchSemaphore(value: 0)
    commandBuffer.addCompletedHandler { _ in completed.signal() }
    commandBuffer.commit()
    guard completed.wait(timeout: .now() + .seconds(20)) == .success else {
        // Throwing reaches the top-level nonzero exit. No other case is run, the
        // attachment is not inspected, and no partial report is published.
        throw OracleError("\(definition.id): GPU completion timed out after 20 seconds; submitted work was not cancelled")
    }
    try require(commandBuffer.status == .completed && commandBuffer.error == nil,
                "\(definition.id): Metal execution failed (status \(commandBuffer.status.rawValue)): \(String(describing: commandBuffer.error))")

    var writebacks = [Writeback]()
    var allocations = [AllocationResult]()
    for (index, attachment) in fixture.attachments.enumerated() {
        // A discarded attachment's bytes disappear with the pass: no readback,
        // no writeback and no allocation observation. Reporting it would
        // present a comparison the v19 rule does not admit
        // (`research/docs/23` §3.6).
        if attachment.store != "store" {
            continue
        }
        guard let expected = attachment.expected else {
            throw OracleError("\(definition.id): a stored attachment needs an expectation")
        }
        var observed = Data(count: attachment.width * attachment.height * 4)
        observed.withUnsafeMutableBytes { bytes in
            if let destination = bytes.baseAddress {
                targets[index].getBytes(destination,
                                        bytesPerRow: attachment.width * 4,
                                        from: MTLRegionMake2D(0, 0, attachment.width, attachment.height),
                                        mipmapLevel: 0)
            }
        }
        // The comparison walks the attachment's own image and steps over the
        // texels the case does not claim (`research/docs/23` §3.3, v33): a
        // wildcard texel's four bytes are neither compared nor named, because
        // the fixture said in advance that whatever lands there is undefined.
        // Every other byte still has to equal the reviewed expectation, the
        // first claimed byte that differs is reported with its offset, and the
        // writeback and allocation below still carry the measured bytes.
        try require(observed.count == expected.count,
                    "\(definition.id): attachment \(index) read back \(observed.count) bytes "
                    + "against the reviewed expectation's \(expected.count)")
        var differing: Int?
        for (offset, pair) in zip(expected, observed).enumerated() {
            if attachment.wildcardBytes.contains(offset) {
                continue
            }
            if pair.0 != pair.1 {
                differing = offset
                break
            }
        }
        if let differing {
            throw OracleError("\(definition.id): attachment \(index) bytes \(hex(observed)) "
                              + "do not match the reviewed expectation \(hex(expected)) "
                              + "(first differing byte at offset \(differing))")
        }
        // One writeback and one allocation per attachment, both the
        // attachment's own texels.
        writebacks.append(Writeback(allocation: attachment.allocation, view: attachment.view,
                                    offset: 0, bytes_hex: hex(observed)))
        allocations.append(AllocationResult(allocation: attachment.allocation,
                                             bytes_hex: hex(observed)))
    }
    // A stored depth surface reports its texels through the same channel
    // (`research/docs/23` §3.3, v43): one writeback for the depth view at
    // offset zero and one allocation observation, in the surface's own
    // `depth32float` texel layout. It is appended after the colour
    // attachments, which is the report's `(allocation, view)` order for this
    // shape — a case's colour landing (900/910) sorts before its depth landing
    // (940/950). The comparison reports the first differing byte and both
    // sides, the same shape the colour readback above uses.
    if let depth = fixture.depth, let store = depth.store {
        // A resolving pass observes the single-sample landing its resolve
        // wrote; every non-resolving stored surface is read back from its own
        // texture (`research/docs/23` §3.3, v43/v57c).
        let readbackTexture = definition.depth_resolve != nil
            ? depthResolveTarget : depthTarget
        guard let texture = readbackTexture else {
            throw OracleError("\(definition.id): the stored depth attachment left the pass")
        }
        var observed = Data(count: depth.width * depth.height * 4)
        observed.withUnsafeMutableBytes { bytes in
            if let destination = bytes.baseAddress {
                texture.getBytes(destination,
                                 bytesPerRow: depth.width * 4,
                                 from: MTLRegionMake2D(0, 0, depth.width, depth.height),
                                 mipmapLevel: 0)
            }
        }
        try require(observed.count == store.expected.count,
                    "\(definition.id): the depth attachment read back \(observed.count) bytes "
                    + "against the reviewed expectation's \(store.expected.count)")
        var differing: Int?
        for (offset, pair) in zip(store.expected, observed).enumerated() {
            if pair.0 != pair.1 {
                differing = offset
                break
            }
        }
        if let differing {
            throw OracleError("\(definition.id): depth texels \(hex(observed)) do not match "
                              + "the reviewed expectation \(hex(store.expected)) "
                              + "(first differing byte at offset \(differing))")
        }
        writebacks.append(Writeback(allocation: store.allocation, view: store.view,
                                    offset: 0, bytes_hex: hex(observed)))
        allocations.append(AllocationResult(allocation: store.allocation,
                                             bytes_hex: hex(observed)))
    }
    // A stored stencil surface reports its texels through the same channel
    // (`research/docs/23` §3.3, v49): one writeback for the stencil view at
    // offset zero and one allocation observation, one byte per `stencil8`
    // texel rather than the depth sibling's four — so the readback is
    // `width * height` bytes with the surface's own `width` as the row pitch.
    // It is appended after the colour attachments and after the depth surface,
    // which is the report's `(allocation, view)` order for this shape: the
    // case's colour landing (900/910) sorts before its stencil landing
    // (940/951). The comparison reports the first differing byte and both
    // sides, the same shape the colour and depth readbacks above use.
    if let stencil = fixture.stencil, let store = stencil.store {
        // A resolving pass observes the single-sample landing its resolve
        // wrote; every non-resolving stored surface is read back from its own
        // texture (`research/docs/23` §3.3, v49/v60).
        let readbackTexture = definition.stencil_resolve != nil
            ? stencilResolveTarget : stencilTarget
        guard let texture = readbackTexture else {
            throw OracleError("\(definition.id): the stored stencil attachment left the pass")
        }
        var observed = Data(count: stencil.width * stencil.height)
        observed.withUnsafeMutableBytes { bytes in
            if let destination = bytes.baseAddress {
                texture.getBytes(destination,
                                 bytesPerRow: stencil.width,
                                 from: MTLRegionMake2D(0, 0, stencil.width, stencil.height),
                                 mipmapLevel: 0)
            }
        }
        try require(observed.count == store.expected.count,
                    "\(definition.id): the stencil attachment read back \(observed.count) bytes "
                    + "against the reviewed expectation's \(store.expected.count)")
        var differing: Int?
        for (offset, pair) in zip(store.expected, observed).enumerated() {
            if pair.0 != pair.1 {
                differing = offset
                break
            }
        }
        if let differing {
            throw OracleError("\(definition.id): stencil texels \(hex(observed)) do not match "
                              + "the reviewed expectation \(hex(store.expected)) "
                              + "(first differing byte at offset \(differing))")
        }
        writebacks.append(Writeback(allocation: store.allocation, view: store.view,
                                    offset: 0, bytes_hex: hex(observed)))
        allocations.append(AllocationResult(allocation: store.allocation,
                                             bytes_hex: hex(observed)))
    }
    return CaseResult(id: definition.id, completion: "CompletedVisible",
                      writebacks: writebacks, allocations: allocations)
}

/// The milestone's own render fixture, constructed in code.
///
/// `--suite conformance/suite-v13.json` reaches `runRenderCase` through
/// `capture`; this is the same fixture without a suite, and it is the one
/// command the provider's render-bit flip condition refers to
/// (`conformance/RENDER-CAPTURE.md` §5). It resolves the reviewed pin
/// (`shaders/render_offscreen_2x2.metal`) against the working directory, so it
/// runs from `conformance/` exactly like that section's command does, and it
/// fails unless all four attachment texels read back as the fragment's
/// `40 80 c0 ff` instead of the `fe` clear sentinel.
@available(macOS 11.0, *)
private func renderSelfTest() throws -> CaseResult {
    let reviewed = reviewedRenderModule()
    let definition = RenderCaseDefinition(
        id: "render_offscreen_2x2",
        declaring_case: "",
        vertex_entry: reviewed.vertex_entry,
        fragment_entry: reviewed.fragment_entry,
        metal: reviewed.metal,
        vertices: 3,
        viewport: [0, 0, 2, 2],
        scissor: nil,
        instance_count: nil,
        base_vertex: nil,
        // The `vertex_id` shape: positions come from the vertex index, so the
        // case declares no layout, no stream and no index buffer.
        vertex_layout: nil,
        vertex_buffers: nil,
        indices: nil,
        attachment: RenderAttachmentDefinition(
            allocation: 900, view: 910, format: "rgba8_unorm",
            width: 2, height: 2, load: "clear", store: "store",
            clear_hex: "fefefefe", initial_hex: nil, expected_hex: nil),
        attachments: nil,
        expected_hex: "4080c0ff4080c0ff4080c0ff4080c0ff",
        coverage: nil,
        multisample: nil,
        depth_resolve: nil,
        requires_depth_resolve_filter: nil,
        stencil_resolve: nil,
        requires_stencil_resolve_filter: nil,
        wildcard_texels: nil,
        // The `vertex_id` shape is depth-less, the semantics every pre-v36
        // case has (`research/docs/23` §3.3, v36).
        depth: nil,
        depth_test: nil,
        cull: nil,
        blend: nil,
        // Every pre-v47 shape is stencil-less (`research/docs/23` §3.3, v47).
        stencil: nil,
        stencil_test: nil,
        // The self-test runs on this rail by construction; the marker is the
        // same one suite-v13 names for it.
        capture_rails: ["native-metal"])
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
    // No stream to read: the shape declares none, and the vertex stage reads
    // its positions from `vertex_id`.
    let fixture = try validateRenderCase(definition, root: root)
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw OracleError("No default Metal device is available; the render self-test requires an Apple silicon Mac")
    }
    let eligibility = assessDevice(device)
    try require(eligibility.eligible,
                "This oracle requires a named Apple silicon GPU with nonuniform threadgroups and unified memory")
    guard let queue = device.makeCommandQueue() else {
        throw OracleError("Cannot create a Metal command queue")
    }
    diagnostic("native render self-test: device=\(device.name) platform=\(eligibility.platform)")
    return try runRenderCase(fixture, device: device, queue: queue)
}

/// The present milestone's own fixture, constructed in code.
///
/// This is the byte-level present equivalent the Swift oracle can express
/// (`research/docs/24` §5.1, §6 Step 7): the present target is a 2x2
/// `rgba8Unorm` texture preset with the `fefefefe` sentinel, the reviewed
/// fragment draws over it through a `Load` (keeping the sentinel as the
/// previous contents), and the readback has to be the fragment's
/// `40 80 c0 ff` texel rather than the sentinel. The acquire/present counts
/// are provider-only observations (§5.1, §5.3), so the oracle reports the same
/// `writebacks`/`allocations` shape the render self-test does and leaves the
/// count assertion to the provider backend.
@available(macOS 11.0, *)
private func presentSelfTest() throws -> CaseResult {
    let reviewed = reviewedRenderModule()
    // The sentinel is one texel replicated across the 2x2 target, exactly as
    // `InitialState::Sentinel` presets the whole present target.
    let sentinel = Data(repeating: 0xfe, count: 16)
    let definition = RenderCaseDefinition(
        id: "present_offscreen_2x2",
        declaring_case: "",
        vertex_entry: reviewed.vertex_entry,
        fragment_entry: reviewed.fragment_entry,
        metal: reviewed.metal,
        vertices: 3,
        viewport: [0, 0, 2, 2],
        scissor: nil,
        instance_count: nil,
        base_vertex: nil,
        // The present equivalent replays the `vertex_id` shape, so it declares
        // no vertex input either.
        vertex_layout: nil,
        vertex_buffers: nil,
        indices: nil,
        attachment: RenderAttachmentDefinition(
            allocation: 900, view: 910, format: "rgba8_unorm",
            width: 2, height: 2, load: "load", store: "store",
            clear_hex: nil, initial_hex: hex(sentinel), expected_hex: nil),
        attachments: nil,
        expected_hex: "4080c0ff4080c0ff4080c0ff4080c0ff",
        coverage: nil,
        multisample: nil,
        depth_resolve: nil,
        requires_depth_resolve_filter: nil,
        stencil_resolve: nil,
        requires_stencil_resolve_filter: nil,
        wildcard_texels: nil,
        depth: nil,
        depth_test: nil,
        cull: nil,
        blend: nil,
        // Every pre-v47 shape is stencil-less (`research/docs/23` §3.3, v47).
        stencil: nil,
        stencil_test: nil,
        // The self-test is this rail's own check; it runs directly rather than
        // through a suite marker, so the marker only has to name this rail.
        capture_rails: ["native-metal"])
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
    let fixture = try validateRenderCase(definition, root: root)
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw OracleError("No default Metal device is available; the present self-test requires an Apple silicon Mac")
    }
    let eligibility = assessDevice(device)
    try require(eligibility.eligible,
                "This oracle requires a named Apple silicon GPU with nonuniform threadgroups and unified memory")
    guard let queue = device.makeCommandQueue() else {
        throw OracleError("Cannot create a Metal command queue")
    }
    diagnostic("native present self-test: device=\(device.name) platform=\(eligibility.platform)")
    return try runRenderCase(fixture, device: device, queue: queue)
}

/// The vertex-input milestone's own fixture, constructed in code.
///
/// This is the one-device check the native provider's vertex-input bits point
/// at (`conformance/RENDER-CAPTURE.md` §8): the reviewed indexed module, the
/// fixture's `float32x2` stream of four NDC corners and its six `uint16`
/// indices, bound through an `MTLVertexDescriptor` and drawn with
/// `drawIndexedPrimitives` into the same 2x2 `rgba8Unorm` attachment the render
/// self-test uses. The stream and index bytes are spelled here exactly as a
/// suite spells them in the view's own `initial_hex` (`research/docs/23` §3.6),
/// so the self-test reaches `runRenderCase` through the same validation a suite
/// capture does. It fails unless **all four texels read back as the reviewed
/// fragment's `40 80 c0 ff`** instead of the `fe` clear sentinel, and unless the
/// report's `writebacks`/`allocations` name the attachment view — the same
/// falsifiability rule as `--render-selftest`, reached through the caller-held
/// streams this increment adds. The pass counts as observed only once a runner
/// reusing `conformance/run_native.py::validate_vertex_selftest` prints
/// `vertex_selftest: PASS (4080c0ff)`.
@available(macOS 11.0, *)
private func vertexSelfTest() throws -> CaseResult {
    let reviewed = reviewedIndexedModule()
    // The four NDC corners, `float32x2` little-endian: (-1,-1), (1,-1), (-1,1),
    // (1,1). The same 32 bytes `render.rs`'s fixture builds.
    let vertices = Data([
        0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x80, 0xbf,
        0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0xbf,
        0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x80, 0x3f,
        0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f,
    ])
    // The six `uint16` indices (0,1,2) and (2,1,3): the two triangles that
    // cover the whole square, 12 bytes.
    let indices = Data([
        0x00, 0x00, 0x01, 0x00, 0x02, 0x00,
        0x02, 0x00, 0x01, 0x00, 0x03, 0x00,
    ])
    let definition = RenderCaseDefinition(
        id: "vertex_quad_indexed_2x2",
        declaring_case: "",
        vertex_entry: reviewed.vertex_entry,
        fragment_entry: reviewed.fragment_entry,
        metal: reviewed.metal,
        // `vertices` is the index count in the indexed shape.
        vertices: 6,
        viewport: [0, 0, 2, 2],
        scissor: nil,
        instance_count: nil,
        base_vertex: nil,
        vertex_layout: RenderVertexLayoutDefinition(buffers: reviewed.buffers ?? []),
        // The stream and index views, spelled exactly as a suite spells them:
        // each view carries its own bytes (`research/docs/23` §3.6), which is
        // the same shape `validateRenderCase` reads out of a suite's render
        // case. Both start at their allocation's first byte, like the reviewed
        // fixture's do.
        vertex_buffers: [RenderVertexBufferDefinition(allocation: 940, view: 950, offset: 0,
                                                      length: UInt64(vertices.count),
                                                      initial_hex: hex(vertices))],
        indices: RenderIndexBufferDefinition(allocation: 960, view: 970, offset: 0,
                                             length: UInt64(indices.count),
                                             initial_hex: hex(indices),
                                             format: "uint16"),
        attachment: RenderAttachmentDefinition(
            allocation: 900, view: 910, format: "rgba8_unorm",
            width: 2, height: 2, load: "clear", store: "store",
            clear_hex: "fefefefe", initial_hex: nil, expected_hex: nil),
        attachments: nil,
        expected_hex: "4080c0ff4080c0ff4080c0ff4080c0ff",
        coverage: nil,
        multisample: nil,
        depth_resolve: nil,
        requires_depth_resolve_filter: nil,
        stencil_resolve: nil,
        requires_stencil_resolve_filter: nil,
        wildcard_texels: nil,
        depth: nil,
        depth_test: nil,
        cull: nil,
        blend: nil,
        // Every pre-v47 shape is stencil-less (`research/docs/23` §3.3, v47).
        stencil: nil,
        stencil_test: nil,
        // The self-test runs on this rail by construction; the marker is the
        // same one a suite would name for it.
        capture_rails: ["native-metal"])
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
    let fixture = try validateRenderCase(definition, root: root)
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw OracleError("No default Metal device is available; the vertex self-test requires an Apple silicon Mac")
    }
    let eligibility = assessDevice(device)
    try require(eligibility.eligible,
                "This oracle requires a named Apple silicon GPU with nonuniform threadgroups and unified memory")
    guard let queue = device.makeCommandQueue() else {
        throw OracleError("Cannot create a Metal command queue")
    }
    diagnostic("native vertex self-test: device=\(device.name) platform=\(eligibility.platform)")
    return try runRenderCase(fixture, device: device, queue: queue)
}

/// The MRT milestone's own fixture, constructed in code.
///
/// This is the one-device check the native provider's dual-output bits point at
/// (`conformance/RENDER-CAPTURE.md` §10): the reviewed dual module, the same
/// indexed quad's `float32x2` stream and `uint16` indices as `--vertex-selftest`,
/// drawn through an `MTLRenderPassDescriptor` whose `colorAttachments[0..2]`
/// each carry their own 2x2 `rgba8Unorm` texture and clear/store actions. The
/// fragment writes `4080c0ff` to location 0 and `ff8040c0` to location 1, and
/// the report fails unless each attachment reads back its own texel instead of
/// the `fefefefe` clear sentinel. The pass counts as observed only once a runner
/// reusing `conformance/run_native.py::validate_mrt_selftest` prints
/// `mrt_selftest: PASS (4080c0ff ff8040c0)`.
@available(macOS 11.0, *)
private func mrtSelfTest() throws -> CaseResult {
    let reviewed = reviewedDualModule()
    // The four NDC corners, `float32x2` little-endian: (-1,-1), (1,-1), (-1,1),
    // (1,1). The same 32 bytes the vertex self-test and `render.rs`'s fixture
    // build.
    let vertices = Data([
        0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x80, 0xbf,
        0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0xbf,
        0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x80, 0x3f,
        0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f,
    ])
    // The six `uint16` indices (0,1,2) and (2,1,3): the two triangles that
    // cover the whole square, 12 bytes.
    let indices = Data([
        0x00, 0x00, 0x01, 0x00, 0x02, 0x00,
        0x02, 0x00, 0x01, 0x00, 0x03, 0x00,
    ])
    let definition = RenderCaseDefinition(
        id: "mrt_dual_output_2x2",
        declaring_case: "",
        vertex_entry: reviewed.vertex_entry,
        fragment_entry: reviewed.fragment_entry,
        metal: reviewed.metal,
        // `vertices` is the index count in the indexed shape.
        vertices: 6,
        viewport: [0, 0, 2, 2],
        scissor: nil,
        instance_count: nil,
        base_vertex: nil,
        vertex_layout: RenderVertexLayoutDefinition(buffers: reviewed.buffers ?? []),
        vertex_buffers: [RenderVertexBufferDefinition(allocation: 940, view: 950, offset: 0,
                                                      length: UInt64(vertices.count),
                                                      initial_hex: hex(vertices))],
        indices: RenderIndexBufferDefinition(allocation: 960, view: 970, offset: 0,
                                             length: UInt64(indices.count),
                                             initial_hex: hex(indices),
                                             format: "uint16"),
        attachment: nil,
        // Two attachments, in location order: allocation 900/view 910 is
        // location 0, allocation 901/view 911 is location 1. Both are cleared
        // with the sentinel, which neither location's texel equals.
        attachments: [
            RenderAttachmentDefinition(
                allocation: 900, view: 910, format: "rgba8_unorm",
                width: 2, height: 2, load: "clear", store: "store",
                clear_hex: "fefefefe", initial_hex: nil,
                expected_hex: "4080c0ff4080c0ff4080c0ff4080c0ff"),
            RenderAttachmentDefinition(
                allocation: 901, view: 911, format: "rgba8_unorm",
                width: 2, height: 2, load: "clear", store: "store",
                clear_hex: "fefefefe", initial_hex: nil,
                expected_hex: "ff8040c0ff8040c0ff8040c0ff8040c0"),
        ],
        // Location 0 first, then location 1: the fixture's own byte strings,
        // spelled per attachment the way a suite's MRT case does.
        expected_hex: nil,
        coverage: nil,
        multisample: nil,
        depth_resolve: nil,
        requires_depth_resolve_filter: nil,
        stencil_resolve: nil,
        requires_stencil_resolve_filter: nil,
        wildcard_texels: nil,
        depth: nil,
        depth_test: nil,
        cull: nil,
        blend: nil,
        // Every pre-v47 shape is stencil-less (`research/docs/23` §3.3, v47).
        stencil: nil,
        stencil_test: nil,
        // The self-test runs on this rail by construction; the marker is the
        // same one a suite would name for it.
        capture_rails: ["native-metal"])
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
    let fixture = try validateRenderCase(definition, root: root)
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw OracleError("No default Metal device is available; the MRT self-test requires an Apple silicon Mac")
    }
    let eligibility = assessDevice(device)
    try require(eligibility.eligible,
                "This oracle requires a named Apple silicon GPU with nonuniform threadgroups and unified memory")
    guard let queue = device.makeCommandQueue() else {
        throw OracleError("Cannot create a Metal command queue")
    }
    diagnostic("native MRT self-test: device=\(device.name) platform=\(eligibility.platform)")
    return try runRenderCase(fixture, device: device, queue: queue)
}

/// The heap milestone's own fixture, constructed in code.
///
/// This is the one-device heap check (`research/docs/25` §6 Step 7a): two
/// buffers allocated from one `MTLHeap`, their offsets recorded, the reviewed
/// `copy_word` kernel run across the pair, and the write buffer read back. It
/// fails unless both buffers are heap-backed and share one heap with
/// non-overlapping byte ranges, and the write buffer's first word equals the
/// reviewed `fefefefe` rather than the `ffffffff` sentinel it was preset with.
/// `MTLHeap` does not promise to honour a requested offset, so this selftest
/// only asserts "same heap + non-overlap + correct bytes"; the suite's explicit
/// offset semantics belong to the provider-side slab + sub-range equivalent.
@available(macOS 11.0, *)
private func heapSelfTest() throws -> HeapSelfTestReport {
    let program = try reviewedProgram("copy_word")
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
    let source = try loadProgram(program, root: root)
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw OracleError("No default Metal device is available; the heap self-test requires an Apple silicon Mac")
    }
    let eligibility = assessDevice(device)
    try require(eligibility.eligible,
                "This oracle requires a named Apple silicon GPU with nonuniform threadgroups and unified memory")
    guard let queue = device.makeCommandQueue() else {
        throw OracleError("Cannot create a Metal command queue")
    }
    diagnostic("native heap self-test: device=\(device.name) platform=\(eligibility.platform)")

    let heapDescriptor = MTLHeapDescriptor()
    heapDescriptor.size = 512
    heapDescriptor.storageMode = .shared
    guard let heap = device.makeHeap(descriptor: heapDescriptor) else {
        throw OracleError("heap self-test: cannot allocate the heap")
    }
    guard let readBuffer = heap.makeBuffer(length: 16, options: .storageModeShared) else {
        throw OracleError("heap self-test: cannot allocate the read buffer")
    }
    guard let writeBuffer = heap.makeBuffer(length: 12, options: .storageModeShared) else {
        throw OracleError("heap self-test: cannot allocate the write buffer")
    }
    let readInitial = Data(repeating: 0xfe, count: 16)
    let writeInitial = Data(repeating: 0xff, count: 12)
    readInitial.withUnsafeBytes { bytes in
        if let source = bytes.baseAddress {
            readBuffer.contents().copyMemory(from: source, byteCount: readInitial.count)
        }
    }
    writeInitial.withUnsafeBytes { bytes in
        if let source = bytes.baseAddress {
            writeBuffer.contents().copyMemory(from: source, byteCount: writeInitial.count)
        }
    }
    // The placement observation: both buffers are heap-backed, share one heap,
    // and occupy non-overlapping byte ranges. `nil === nil` is true, so the
    // identity check only runs after both heap references are non-nil.
    try require(readBuffer.heap != nil && writeBuffer.heap != nil,
                "heap self-test: a buffer was not heap-backed")
    try require(readBuffer.heap === writeBuffer.heap,
                "heap self-test: the two buffers are not in the same heap")
    // `MTLResource.heapOffset` is the byte offset a heap-backed resource was
    // placed at; `MTLBuffer.offset` does not exist.
    let readOffset = readBuffer.heapOffset
    let writeOffset = writeBuffer.heapOffset
    diagnostic("native heap self-test: read_offset=\(readOffset) write_offset=\(writeOffset)")
    if readOffset == writeOffset {
        // The Apple Paravirtual device reports both heap offsets as 0, so its
        // placement cannot be observed from `heapOffset`; the byte check below
        // is this rail's evidence, and the explicit-offset contract lives on
        // the provider side (`crates/metal-api-native/src/heap.rs`), which
        // binds one slab at the declared offsets.
        diagnostic("native heap self-test: heap offsets are not reported; "
                    + "the reviewed kernel's readback is the evidence")
    } else {
        try require(readOffset + 16 <= writeOffset || writeOffset + 12 <= readOffset,
                    "heap self-test: the two heap ranges overlap")
    }

    let library = try device.makeLibrary(source: source, options: nil)
    guard let function = library.makeFunction(name: "copy_word") else {
        throw OracleError("heap self-test: copy_word function was not found")
    }
    let pipeline = try device.makeComputePipelineState(function: function)
    guard let commandBuffer = queue.makeCommandBuffer() else {
        throw OracleError("heap self-test: cannot create a command buffer")
    }
    try require(commandBuffer.retainedReferences,
                "heap self-test: command buffer does not retain resources")
    commandBuffer.label = "native oracle: heap selftest"
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        throw OracleError("heap self-test: cannot create a compute encoder")
    }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(readBuffer, offset: 0, index: 0)
    encoder.setBuffer(writeBuffer, offset: 0, index: 1)
    encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    encoder.endEncoding()
    let completed = DispatchSemaphore(value: 0)
    commandBuffer.addCompletedHandler { _ in completed.signal() }
    commandBuffer.commit()
    guard completed.wait(timeout: .now() + .seconds(20)) == .success else {
        throw OracleError("heap self-test: GPU completion timed out after 20 seconds; submitted work was not cancelled")
    }
    try require(commandBuffer.status == .completed && commandBuffer.error == nil,
                "heap self-test: Metal execution failed (status \(commandBuffer.status.rawValue)): \(String(describing: commandBuffer.error))")

    let copied = Data(bytes: writeBuffer.contents(), count: 4)
    let expected = Data(repeating: 0xfe, count: 4)
    try require(copied == expected,
                "heap self-test: copied word \(hex(copied)) does not match the reviewed expectation \(hex(expected))")
    let writeback = Writeback(allocation: 920, view: 930, offset: 0, bytes_hex: hex(copied))
    let allocations = [
        AllocationResult(allocation: 900,
                         bytes_hex: hex(Data(bytes: readBuffer.contents(), count: 16))),
        AllocationResult(allocation: 920,
                         bytes_hex: hex(Data(bytes: writeBuffer.contents(), count: 12))),
    ]
    return HeapSelfTestReport(id: "heap_placement_copy_word", completion: "CompletedVisible",
                              writebacks: [writeback], allocations: allocations,
                              device: device.name, platform: eligibility.platform)
}

/// One reviewed pass of the depth-resolve self-test.
///
/// The reviewed depth pair module draws the caller's edge geometry through a
/// four-sample `depth32float` raster cleared to one, a `less` test with writes
/// on, and a `.multisampleResolve` store action whose `depthResolveFilter` is
/// the caller's. The single-sample shared landing is read back with `getBytes`
/// and returned: one `float32` per texel in memory order, 64 bytes. The colour
/// side mirrors `runRenderCase`'s multisample pair but is never read back —
/// the depth landing is the whole observation.
@available(macOS 11.0, *)
private func depthResolveSelftestPass(_ fixture: ValidatedRender, device: MTLDevice,
                                      queue: MTLCommandQueue,
                                      filter: MTLMultisampleDepthResolveFilter,
                                      name: String) throws -> Data {
    let definition = fixture.definition
    guard let attachment = fixture.attachments.first, let depth = fixture.depth,
          let multisample = definition.multisample else {
        throw OracleError("\(definition.id): the depth-resolve self-test needs one colour "
                          + "attachment, one depth attachment and the four-sample raster")
    }
    let samples = Int(multisample.sample_count)

    // The colour pair the reviewed multisample raster draws through. Its texels
    // are not observed, so both surfaces stay private, exactly as
    // `runRenderCase` keeps the four-sample half private.
    let colourDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: attachment.pixelFormat,
        width: attachment.width,
        height: attachment.height,
        mipmapped: false)
    colourDescriptor.textureType = .type2DMultisample
    colourDescriptor.sampleCount = samples
    colourDescriptor.usage = .renderTarget
    colourDescriptor.storageMode = .private
    guard let colourMSAA = device.makeTexture(descriptor: colourDescriptor) else {
        throw OracleError("\(definition.id): cannot allocate the multisample colour attachment")
    }
    colourMSAA.label = "native oracle: \(definition.id) colour msaa"
    let colourResolveDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: attachment.pixelFormat,
        width: attachment.width,
        height: attachment.height,
        mipmapped: false)
    colourResolveDescriptor.usage = .renderTarget
    colourResolveDescriptor.storageMode = .private
    guard let colourResolve = device.makeTexture(descriptor: colourResolveDescriptor) else {
        throw OracleError("\(definition.id): cannot allocate the colour resolve target")
    }
    colourResolve.label = "native oracle: \(definition.id) colour resolve"

    // The depth half: a private four-sample surface the raster writes, and the
    // single-sample shared landing the resolve writes into — the same split
    // `runRenderCase` states for the stored depth resolve shape
    // (`research/docs/23` §3.3, v57c), and the landing is what the CPU reads
    // back.
    let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .depth32Float,
        width: depth.width,
        height: depth.height,
        mipmapped: false)
    depthDescriptor.textureType = .type2DMultisample
    depthDescriptor.sampleCount = samples
    depthDescriptor.usage = .renderTarget
    depthDescriptor.storageMode = .private
    guard let depthMSAA = device.makeTexture(descriptor: depthDescriptor) else {
        throw OracleError("\(definition.id): cannot allocate the multisample depth attachment")
    }
    depthMSAA.label = "native oracle: \(definition.id) depth msaa"
    let landingDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .depth32Float,
        width: depth.width,
        height: depth.height,
        mipmapped: false)
    landingDescriptor.usage = .renderTarget
    landingDescriptor.storageMode = .shared
    guard let landing = device.makeTexture(descriptor: landingDescriptor) else {
        throw OracleError("\(definition.id): cannot allocate the depth resolve landing")
    }
    landing.label = "native oracle: \(definition.id) depth resolve \(name)"

    // The pipeline states the same shape `runRenderCase` builds for the depth
    // pair's multisampled raster: one `rgba8Unorm` location, the depth format
    // and the four-sample raster count.
    let library = try device.makeLibrary(source: fixture.source, options: nil)
    guard let vertexFunction = library.makeFunction(name: definition.vertex_entry) else {
        throw OracleError("\(definition.id): vertex entry \(definition.vertex_entry) was not found")
    }
    guard let fragmentFunction = library.makeFunction(name: definition.fragment_entry) else {
        throw OracleError("\(definition.id): fragment entry \(definition.fragment_entry) was not found")
    }
    let pipelineDescriptor = MTLRenderPipelineDescriptor()
    pipelineDescriptor.label = "native oracle: \(definition.id)"
    pipelineDescriptor.vertexFunction = vertexFunction
    pipelineDescriptor.fragmentFunction = fragmentFunction
    let vertexDescriptor = MTLVertexDescriptor()
    for stream in fixture.vertexStreams {
        guard let layout = vertexDescriptor.layouts[stream.binding] else {
            throw OracleError("\(definition.id): the vertex descriptor has no layout "
                              + "\(stream.binding)")
        }
        layout.stride = Int(stream.stride)
        layout.stepFunction = .perVertex
        for attribute in stream.attributes {
            guard let format = vertexFormat(attribute.format) else {
                throw OracleError("\(definition.id): unsupported vertex attribute format "
                                  + attribute.format)
            }
            guard let target = vertexDescriptor.attributes[Int(attribute.location)] else {
                throw OracleError("\(definition.id): the vertex descriptor has no attribute "
                                  + "\(attribute.location)")
            }
            target.format = format
            target.offset = Int(attribute.offset)
            target.bufferIndex = stream.binding
        }
    }
    pipelineDescriptor.vertexDescriptor = vertexDescriptor
    pipelineDescriptor.colorAttachments[0].pixelFormat = attachment.pixelFormat
    pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
    pipelineDescriptor.rasterSampleCount = samples
    let pipeline = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)

    // The pass opens the colour and depth halves with clears and resolves both
    // on store; the depth resolve filter is the one this pass names.
    let pass = MTLRenderPassDescriptor()
    guard let colour = pass.colorAttachments[0], let depthAttachment = pass.depthAttachment else {
        throw OracleError("\(definition.id): cannot reach the self-test's attachments")
    }
    colour.texture = colourMSAA
    colour.resolveTexture = colourResolve
    colour.loadAction = .clear
    colour.clearColor = MTLClearColor(red: attachment.clearComponents[0],
                                      green: attachment.clearComponents[1],
                                      blue: attachment.clearComponents[2],
                                      alpha: attachment.clearComponents[3])
    colour.storeAction = .multisampleResolve
    depthAttachment.texture = depthMSAA
    depthAttachment.loadAction = .clear
    depthAttachment.clearDepth = depth.clearDepth
    depthAttachment.storeAction = .multisampleResolve
    depthAttachment.resolveTexture = landing
    depthAttachment.depthResolveFilter = filter

    guard let commandBuffer = queue.makeCommandBuffer() else {
        throw OracleError("\(definition.id): cannot create a command buffer")
    }
    try require(commandBuffer.retainedReferences,
                "\(definition.id): command buffer does not retain resources")
    commandBuffer.label = "native oracle: \(definition.id) depth resolve \(name)"
    guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
        throw OracleError("\(definition.id): cannot create a render encoder")
    }
    encoder.setRenderPipelineState(pipeline)
    let depthStencilDescriptor = MTLDepthStencilDescriptor()
    depthStencilDescriptor.depthCompareFunction = depth.isLess ? .less : .always
    depthStencilDescriptor.isDepthWriteEnabled = depth.write
    guard let state = device.makeDepthStencilState(descriptor: depthStencilDescriptor) else {
        throw OracleError("\(definition.id): cannot create the depth-stencil state")
    }
    encoder.setDepthStencilState(state)
    encoder.setViewport(MTLViewport(originX: 0, originY: 0,
                                    width: Double(attachment.width),
                                    height: Double(attachment.height),
                                    znear: 0, zfar: 1))
    // The streams are bound at the same indices the descriptor names, and they
    // stay alive until the command buffer has completed (the buffers array is
    // released after the readback below), the same keep-alive `runRenderCase`
    // maintains.
    var streamBuffers = [MTLBuffer]()
    for stream in fixture.vertexStreams {
        let buffer = try makeStreamBuffer(device: device, id: definition.id,
                                          offset: stream.offset, bytes: stream.bytes)
        streamBuffers.append(buffer)
    }
    for (stream, buffer) in zip(fixture.vertexStreams, streamBuffers) {
        encoder.setVertexBuffer(buffer,
                                offset: try hostOffset(stream.offset, id: definition.id),
                                index: stream.binding)
    }
    if let indexStream = fixture.indexStream {
        let indexBuffer = try makeStreamBuffer(device: device, id: definition.id,
                                               offset: indexStream.offset,
                                               bytes: indexStream.bytes)
        streamBuffers.append(indexBuffer)
        encoder.drawIndexedPrimitives(type: .triangle,
                                      indexCount: Int(indexStream.indexCount),
                                      indexType: indexStream.format.metal,
                                      indexBuffer: indexBuffer,
                                      indexBufferOffset: try hostOffset(indexStream.offset,
                                                                        id: definition.id),
                                      instanceCount: 1)
    }
    encoder.endEncoding()
    let completed = DispatchSemaphore(value: 0)
    commandBuffer.addCompletedHandler { _ in completed.signal() }
    commandBuffer.commit()
    guard completed.wait(timeout: .now() + .seconds(20)) == .success else {
        throw OracleError("\(definition.id): GPU completion timed out after 20 seconds; submitted work was not cancelled")
    }
    try require(commandBuffer.status == .completed && commandBuffer.error == nil,
                "\(definition.id): Metal execution failed (status \(commandBuffer.status.rawValue)): \(String(describing: commandBuffer.error))")

    var observed = Data(count: depth.width * depth.height * 4)
    observed.withUnsafeMutableBytes { bytes in
        if let destination = bytes.baseAddress {
            landing.getBytes(destination,
                             bytesPerRow: depth.width * 4,
                             from: MTLRegionMake2D(0, 0, depth.width, depth.height),
                             mipmapLevel: 0)
        }
    }
    return observed
}

/// The v51 edge pair both resolve self-tests share, constructed in code.
///
/// The reviewed depth pair module draws the v51 edge geometry — a near
/// triangle at z = 0.5 covering NDC x <= 0.25 and a far triangle at z = 0.9
/// covering everything, both tinted red — through a four-sample
/// `depth32float` raster cleared to 1.0 with a `less` test and writes on.
/// Both triangles carry the same red tint, so the colour landing cannot vary
/// between them and the resolved depth (or stencil) landing is the whole
/// observation. The definition is the reviewed depth-pair shape
/// `validateRenderCase` pins, so both self-tests get the module hash, stream
/// and index bytes, attachment extent and clear depth from the one validated
/// fixture; the stencil sibling supplies its own stencil surface and state
/// beside it.
@available(macOS 11.0, *)
private func resolvePairFixture(id: String) throws -> ValidatedRender {
    let reviewed = reviewedDepthModule()
    // The v51 edge geometry the v57d Min/Max pair pins (`research/docs/23`
    // §3.3), spelled exactly as the suite's view bytes: one stride-32 stream
    // whose two records carry position `float32x3` at offset 0 and tint
    // `float32x4` at offset 16, with four padding bytes between them.
    let vertices = try decodeHex(
        "0000803e000080bf0000003f000000000000803f00000000000000000000803f"
        + "0000803e000040400000003f000000000000803f00000000000000000000803f"
        + "000040c0000080bf0000003f000000000000803f00000000000000000000803f"
        + "000080bf000080bf6666663f000000000000803f00000000000000000000803f"
        + "00004040000080bf6666663f000000000000803f00000000000000000000803f"
        + "000080bf000040406666663f000000000000803f00000000000000000000803f",
        context: "\(id) vertex stream")
    // The six `uint16` indices (0,1,2) and (3,4,5): the near edge triangle
    // then the full-screen far triangle, 12 bytes.
    let indices = try decodeHex("000001000200030004000500",
                                context: "\(id) index buffer")
    // Both triangles' reviewed tint: red, one `rgba8Unorm` texel replicated
    // across the 4x4 attachment. The pass judgements never read the colour
    // bytes, so this expectation is shape-only.
    let redImage = String(repeating: "ff0000ff", count: 16)
    // The min filter's reviewed landing, used only to satisfy the stored
    // depth surface's expectation rule (`research/docs/23` §3.3, v43): the
    // self-tests' own judgements read columns instead of this image.
    let depthExpectation = String(repeating: "0000003f0000003f0000003f6666663f", count: 4)
    let definition = RenderCaseDefinition(
        id: id,
        declaring_case: "",
        vertex_entry: reviewed.vertex_entry,
        fragment_entry: reviewed.fragment_entry,
        metal: reviewed.metal,
        // `vertices` is the index count in the indexed shape.
        vertices: 6,
        viewport: [0, 0, 4, 4],
        scissor: nil,
        instance_count: nil,
        base_vertex: nil,
        vertex_layout: RenderVertexLayoutDefinition(buffers: reviewed.buffers ?? []),
        // The stream and index views, spelled exactly as a suite spells them:
        // each view carries its own bytes (`research/docs/23` §3.6), the same
        // shape `validateRenderCase` reads out of a suite's render case.
        vertex_buffers: [RenderVertexBufferDefinition(allocation: 940, view: 950, offset: 0,
                                                      length: UInt64(vertices.count),
                                                      initial_hex: hex(vertices))],
        indices: RenderIndexBufferDefinition(allocation: 960, view: 970, offset: 0,
                                             length: UInt64(indices.count),
                                             initial_hex: hex(indices),
                                             format: "uint16"),
        attachment: RenderAttachmentDefinition(
            allocation: 900, view: 910, format: "rgba8_unorm",
            width: 4, height: 4, load: "clear", store: "store",
            clear_hex: "11223344", initial_hex: nil, expected_hex: nil),
        attachments: nil,
        expected_hex: redImage,
        coverage: nil,
        multisample: MultisampleDefinition(sample_count: 4),
        // The validation fixture states the filter this rail declares; the
        // passes below state their own filters, so no suite-level gate or mask
        // participates.
        depth_resolve: DepthResolveDefinition(filter: "sample0"),
        requires_depth_resolve_filter: nil,
        stencil_resolve: nil,
        requires_stencil_resolve_filter: nil,
        wildcard_texels: nil,
        depth: DepthAttachmentDefinition(
            format: "depth32float", width: 4, height: 4, load: "clear",
            clear_depth: 1.0, store: "store", allocation: 980, view: 990,
            expected_hex: depthExpectation),
        depth_test: DepthTestDefinition(compare: "less", write: true),
        cull: nil,
        blend: nil,
        stencil: nil,
        stencil_test: nil,
        // The self-tests run on this rail by construction; the marker is the
        // same one a suite would name for them.
        capture_rails: ["native-metal"])
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
    return try validateRenderCase(definition, root: root)
}

/// The depth-resolve milestone's own fixture, constructed in code.
///
/// This is the one-device check for the Min/Max question v57 left open
/// (`research/docs/23` §3.3, v57e): the reviewed depth pair module draws the
/// v51 edge geometry — a near triangle at z = 0.5 covering NDC x <= 0.25 and
/// a far triangle at z = 0.9 covering everything, both tinted red — through a
/// four-sample `depth32float` raster cleared to 1.0 with a `less` test and
/// writes on, three times, once per resolve filter (sample0, min, max). Each
/// pass resolves into its own single-sample shared landing, and the three
/// landings are printed one texel per line so the CI log carries the answer
/// Metal has no queryable mask for. It prints `depth_resolve_selftest: PASS`
/// only when the stable columns are as reviewed — column 0 reads 0.5 and
/// column 3 reads 0.9 for all three filters — and the key column
/// distinguishes the filters: column 2 reads 0.5 for min and 0.9 for max.
/// sample0's column 2 is recorded but not judged, because its landing depends
/// on the rasterizer's sample positions. In particular, a device that reduces
/// min and max to one value is a FAIL — exactly what this check exists to
/// measure, and the reason PASS may not be asserted from min == max.
@available(macOS 11.0, *)
private func depthResolveSelfTest() throws {
    let fixture = try resolvePairFixture(id: "depth_resolve_selftest_4x4")
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw OracleError("No default Metal device is available; the depth-resolve self-test requires an Apple silicon Mac")
    }
    let eligibility = assessDevice(device)
    try require(eligibility.eligible,
                "This oracle requires a named Apple silicon GPU with nonuniform threadgroups and unified memory")
    guard let queue = device.makeCommandQueue() else {
        throw OracleError("Cannot create a Metal command queue")
    }
    diagnostic("native depth-resolve self-test: device=\(device.name) platform=\(eligibility.platform)")

    let filters: [(name: String, filter: MTLMultisampleDepthResolveFilter)] = [
        ("sample0", .sample0), ("min", .min), ("max", .max),
    ]
    var landings = [String: Data]()
    for entry in filters {
        landings[entry.name] = try depthResolveSelftestPass(fixture, device: device,
                                                            queue: queue, filter: entry.filter,
                                                            name: entry.name)
    }

    // The reviewed near (0.5) and far (0.9) `float32` landings in memory
    // order, which the four texel hexes below compare against.
    let nearBytes = Data([0x00, 0x00, 0x00, 0x3f])
    let farBytes = Data([0x66, 0x66, 0x66, 0x3f])
    func texel(_ landing: Data, _ row: Int, _ column: Int) -> Data {
        let start = (row * 4 + column) * 4
        return Data(landing[start..<(start + 4)])
    }
    // One machine-readable line per texel, row-major, each carrying the three
    // filters' landings for that texel. Column 0 (fully near) and column 3
    // (fully far) are the stable columns, and column 2 is the key column the
    // filters are expected to split.
    var output = ""
    for row in 0..<4 {
        for column in 0..<4 {
            let sample0Hex = hex(texel(landings["sample0"]!, row, column))
            let minHex = hex(texel(landings["min"]!, row, column))
            let maxHex = hex(texel(landings["max"]!, row, column))
            output += "depth_resolve_selftest: row=\(row) column=\(column) "
                + "sample0=\(sample0Hex) min=\(minHex) max=\(maxHex)\n"
        }
    }
    // The PASS judgement is the stable columns for every filter plus the key
    // column's split; sample0's column 2 is deliberately absent from it.
    var failures = [String]()
    for row in 0..<4 {
        for entry in filters {
            let column0 = texel(landings[entry.name]!, row, 0)
            if column0 != nearBytes {
                failures.append("column 0 row \(row) \(entry.name)=\(hex(column0)) "
                                + "expected \(hex(nearBytes))")
            }
            let column3 = texel(landings[entry.name]!, row, 3)
            if column3 != farBytes {
                failures.append("column 3 row \(row) \(entry.name)=\(hex(column3)) "
                                + "expected \(hex(farBytes))")
            }
        }
        let minKey = texel(landings["min"]!, row, 2)
        if minKey != nearBytes {
            failures.append("column 2 row \(row) min=\(hex(minKey)) "
                            + "expected \(hex(nearBytes))")
        }
        let maxKey = texel(landings["max"]!, row, 2)
        if maxKey != farBytes {
            failures.append("column 2 row \(row) max=\(hex(maxKey)) "
                            + "expected \(hex(farBytes))")
        }
    }
    if failures.isEmpty {
        output += "depth_resolve_selftest: PASS\n"
        FileHandle.standardOutput.write(Data(output.utf8))
    } else {
        for failure in failures {
            output += "depth_resolve_selftest: FAIL (\(failure))\n"
        }
        FileHandle.standardOutput.write(Data(output.utf8))
        // A device that reduces both filters to one value fails the key
        // column's split, which is the authoritative answer this check
        // exists to collect — so the failure has to propagate rather than
        // print PASS.
        throw OracleError("depth_resolve_selftest: " + failures.joined(separator: "; "))
    }
}

/// One reviewed pass of the stencil-resolve self-test.
///
/// The reviewed depth pair module draws the caller's v51 edge geometry through
/// a four-sample depth-stencil raster cleared to depth 1.0 and stencil 0. The
/// near triangle draws with the reviewed stencil pair's state (equal 0, both
/// masks open, increment-wraps on pass) and writes 1 into the samples it
/// covers; the far triangle draws with a stencil state whose write mask is
/// zero, so it writes depth into the remaining samples without touching their
/// stencil. Every texel whose samples straddle the x = 0.25 edge therefore
/// carries both a stencil-1 near sample and a stencil-0 far sample. The depth
/// attachment resolves with the caller's depth filter and the stencil
/// attachment with the caller's stencil filter, so `.depthResolvedSample`
/// means "the stencil of the sample the depth resolve selected" rather than an
/// unobserved fallback; the single-sample shared stencil landing is read back
/// with `getBytes` and returned, one `stencil8` byte per texel in memory
/// order, 16 bytes.
@available(macOS 11.0, *)
private func stencilResolveSelftestPass(_ fixture: ValidatedRender, device: MTLDevice,
                                        queue: MTLCommandQueue,
                                        depthFilter: MTLMultisampleDepthResolveFilter,
                                        stencilFilter: MTLMultisampleStencilResolveFilter,
                                        name: String) throws -> Data {
    let definition = fixture.definition
    guard let attachment = fixture.attachments.first, let depth = fixture.depth,
          let multisample = definition.multisample else {
        throw OracleError("\(definition.id): the stencil-resolve self-test needs one colour "
                          + "attachment, one depth attachment and the four-sample raster")
    }
    let samples = Int(multisample.sample_count)

    // The colour pair the reviewed multisample raster draws through. Its texels
    // are not observed, so both surfaces stay private, exactly as
    // `runRenderCase` keeps the four-sample half private.
    let colourDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: attachment.pixelFormat,
        width: attachment.width,
        height: attachment.height,
        mipmapped: false)
    colourDescriptor.textureType = .type2DMultisample
    colourDescriptor.sampleCount = samples
    colourDescriptor.usage = .renderTarget
    colourDescriptor.storageMode = .private
    guard let colourMSAA = device.makeTexture(descriptor: colourDescriptor) else {
        throw OracleError("\(definition.id): cannot allocate the multisample colour attachment")
    }
    colourMSAA.label = "native oracle: \(definition.id) colour msaa"
    let colourResolveDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: attachment.pixelFormat,
        width: attachment.width,
        height: attachment.height,
        mipmapped: false)
    colourResolveDescriptor.usage = .renderTarget
    colourResolveDescriptor.storageMode = .private
    guard let colourResolve = device.makeTexture(descriptor: colourResolveDescriptor) else {
        throw OracleError("\(definition.id): cannot allocate the colour resolve target")
    }
    colourResolve.label = "native oracle: \(definition.id) colour resolve"

    // The depth-stencil half: one private four-sample combined
    // `depth32Float_stencil8` surface the raster writes, which both the depth
    // and the stencil attachment name — Metal's own shape for a pass that
    // tests and writes both (`research/docs/23` §3.3, v59). The two
    // single-sample landings the resolves write into are separate textures:
    // the depth landing is not read back — its only job is to keep the depth
    // resolve active so the stencil filter's `.depthResolvedSample` follows
    // the depth filter the pass names — and the stencil landing is what the
    // CPU reads back below.
    let depthStencilDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .depth32Float_stencil8,
        width: depth.width,
        height: depth.height,
        mipmapped: false)
    depthStencilDescriptor.textureType = .type2DMultisample
    depthStencilDescriptor.sampleCount = samples
    depthStencilDescriptor.usage = .renderTarget
    depthStencilDescriptor.storageMode = .private
    guard let depthStencilMSAA = device.makeTexture(descriptor: depthStencilDescriptor) else {
        throw OracleError("\(definition.id): cannot allocate the multisample depth-stencil attachment")
    }
    depthStencilMSAA.label = "native oracle: \(definition.id) depth-stencil msaa"
    let depthLandingDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .depth32Float, width: depth.width, height: depth.height, mipmapped: false)
    depthLandingDescriptor.usage = .renderTarget
    depthLandingDescriptor.storageMode = .private
    guard let depthLanding = device.makeTexture(descriptor: depthLandingDescriptor) else {
        throw OracleError("\(definition.id): cannot allocate the depth resolve landing")
    }
    depthLanding.label = "native oracle: \(definition.id) depth resolve \(name)"

    // The single-sample shared landing the stencil resolve writes into — the
    // same readback shape `runRenderCase` states for the stored stencil
    // surface (`research/docs/23` §3.3, v49), one `stencil8` byte per texel.
    let stencilLandingDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .stencil8, width: depth.width, height: depth.height, mipmapped: false)
    stencilLandingDescriptor.usage = .renderTarget
    stencilLandingDescriptor.storageMode = .shared
    guard let stencilLanding = device.makeTexture(descriptor: stencilLandingDescriptor) else {
        throw OracleError("\(definition.id): cannot allocate the stencil resolve landing")
    }
    stencilLanding.label = "native oracle: \(definition.id) stencil resolve \(name)"

    // The pipeline states the same shape `runRenderCase` builds for the stencil
    // pair's multisampled raster: one `rgba8Unorm` location, the depth and
    // stencil formats and the four-sample raster count.
    let library = try device.makeLibrary(source: fixture.source, options: nil)
    guard let vertexFunction = library.makeFunction(name: definition.vertex_entry) else {
        throw OracleError("\(definition.id): vertex entry \(definition.vertex_entry) was not found")
    }
    guard let fragmentFunction = library.makeFunction(name: definition.fragment_entry) else {
        throw OracleError("\(definition.id): fragment entry \(definition.fragment_entry) was not found")
    }
    let pipelineDescriptor = MTLRenderPipelineDescriptor()
    pipelineDescriptor.label = "native oracle: \(definition.id)"
    pipelineDescriptor.vertexFunction = vertexFunction
    pipelineDescriptor.fragmentFunction = fragmentFunction
    let vertexDescriptor = MTLVertexDescriptor()
    for stream in fixture.vertexStreams {
        guard let layout = vertexDescriptor.layouts[stream.binding] else {
            throw OracleError("\(definition.id): the vertex descriptor has no layout "
                              + "\(stream.binding)")
        }
        layout.stride = Int(stream.stride)
        layout.stepFunction = .perVertex
        for attribute in stream.attributes {
            guard let format = vertexFormat(attribute.format) else {
                throw OracleError("\(definition.id): unsupported vertex attribute format "
                                  + attribute.format)
            }
            guard let target = vertexDescriptor.attributes[Int(attribute.location)] else {
                throw OracleError("\(definition.id): the vertex descriptor has no attribute "
                                  + "\(attribute.location)")
            }
            target.format = format
            target.offset = Int(attribute.offset)
            target.bufferIndex = stream.binding
        }
    }
    pipelineDescriptor.vertexDescriptor = vertexDescriptor
    pipelineDescriptor.colorAttachments[0].pixelFormat = attachment.pixelFormat
    pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float_stencil8
    pipelineDescriptor.stencilAttachmentPixelFormat = .depth32Float_stencil8
    pipelineDescriptor.rasterSampleCount = samples
    let pipeline = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)

    // The pass opens the colour, depth and stencil halves with clears and
    // resolves all three on store; the depth and stencil resolve filters are
    // the ones this pass names.
    let pass = MTLRenderPassDescriptor()
    guard let colour = pass.colorAttachments[0], let depthAttachment = pass.depthAttachment,
          let stencilAttachment = pass.stencilAttachment else {
        throw OracleError("\(definition.id): cannot reach the self-test's attachments")
    }
    colour.texture = colourMSAA
    colour.resolveTexture = colourResolve
    colour.loadAction = .clear
    colour.clearColor = MTLClearColor(red: attachment.clearComponents[0],
                                      green: attachment.clearComponents[1],
                                      blue: attachment.clearComponents[2],
                                      alpha: attachment.clearComponents[3])
    colour.storeAction = .multisampleResolve
    depthAttachment.texture = depthStencilMSAA
    depthAttachment.loadAction = .clear
    depthAttachment.clearDepth = depth.clearDepth
    depthAttachment.storeAction = .multisampleResolve
    depthAttachment.resolveTexture = depthLanding
    depthAttachment.depthResolveFilter = depthFilter
    stencilAttachment.texture = depthStencilMSAA
    stencilAttachment.loadAction = .clear
    stencilAttachment.clearStencil = 0
    stencilAttachment.storeAction = .multisampleResolve
    stencilAttachment.resolveTexture = stencilLanding
    stencilAttachment.stencilResolveFilter = stencilFilter

    // The near draw's state is the reviewed stencil pair's own
    // (`research/docs/23` §3.3, v47): an `equal` test against reference zero
    // with both masks wide open that keeps both failure outcomes and
    // increments-wraps on pass. The far draw's state carries a zero write mask
    // and keep operations, so it tests and writes depth without ever writing
    // stencil — the second primitive the task's construction asks for.
    let nearStencil = MTLStencilDescriptor()
    nearStencil.stencilCompareFunction = .equal
    nearStencil.stencilFailureOperation = .keep
    nearStencil.depthFailureOperation = .keep
    nearStencil.depthStencilPassOperation = .incrementWrap
    nearStencil.readMask = 255
    nearStencil.writeMask = 255
    let nearDescriptor = MTLDepthStencilDescriptor()
    nearDescriptor.depthCompareFunction = .less
    nearDescriptor.isDepthWriteEnabled = true
    nearDescriptor.frontFaceStencil = nearStencil
    nearDescriptor.backFaceStencil = nearStencil
    guard let nearState = device.makeDepthStencilState(descriptor: nearDescriptor) else {
        throw OracleError("\(definition.id): cannot create the near stencil state")
    }
    let farStencil = MTLStencilDescriptor()
    farStencil.stencilCompareFunction = .always
    farStencil.stencilFailureOperation = .keep
    farStencil.depthFailureOperation = .keep
    farStencil.depthStencilPassOperation = .keep
    farStencil.readMask = 0
    farStencil.writeMask = 0
    let farDescriptor = MTLDepthStencilDescriptor()
    farDescriptor.depthCompareFunction = .less
    farDescriptor.isDepthWriteEnabled = true
    farDescriptor.frontFaceStencil = farStencil
    farDescriptor.backFaceStencil = farStencil
    guard let farState = device.makeDepthStencilState(descriptor: farDescriptor) else {
        throw OracleError("\(definition.id): cannot create the far stencil state")
    }

    guard let commandBuffer = queue.makeCommandBuffer() else {
        throw OracleError("\(definition.id): cannot create a command buffer")
    }
    try require(commandBuffer.retainedReferences,
                "\(definition.id): command buffer does not retain resources")
    commandBuffer.label = "native oracle: \(definition.id) stencil resolve \(name)"
    guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
        throw OracleError("\(definition.id): cannot create a render encoder")
    }
    encoder.setRenderPipelineState(pipeline)
    encoder.setViewport(MTLViewport(originX: 0, originY: 0,
                                    width: Double(attachment.width),
                                    height: Double(attachment.height),
                                    znear: 0, zfar: 1))
    // The streams are bound at the same indices the descriptor names, and they
    // stay alive until the command buffer has completed (the buffers array is
    // released after the readback below).
    var streamBuffers = [MTLBuffer]()
    for stream in fixture.vertexStreams {
        let buffer = try makeStreamBuffer(device: device, id: definition.id,
                                          offset: stream.offset, bytes: stream.bytes)
        streamBuffers.append(buffer)
    }
    for (stream, buffer) in zip(fixture.vertexStreams, streamBuffers) {
        encoder.setVertexBuffer(buffer,
                                offset: try hostOffset(stream.offset, id: definition.id),
                                index: stream.binding)
    }
    guard let indexStream = fixture.indexStream else {
        throw OracleError("\(definition.id): the indexed edge pair carries no index stream")
    }
    let indexBuffer = try makeStreamBuffer(device: device, id: definition.id,
                                           offset: indexStream.offset,
                                           bytes: indexStream.bytes)
    streamBuffers.append(indexBuffer)
    let indexBase = try hostOffset(indexStream.offset, id: definition.id)
    // The near edge triangle (indices 0..2) writes stencil under the reviewed
    // state; the full-screen far triangle (indices 3..5) then writes depth
    // under the zero-write-mask state without changing any stencil byte.
    encoder.setDepthStencilState(nearState)
    encoder.setStencilReferenceValue(0)
    encoder.drawIndexedPrimitives(type: .triangle,
                                  indexCount: 3,
                                  indexType: indexStream.format.metal,
                                  indexBuffer: indexBuffer,
                                  indexBufferOffset: indexBase,
                                  instanceCount: 1)
    encoder.setDepthStencilState(farState)
    encoder.drawIndexedPrimitives(type: .triangle,
                                  indexCount: 3,
                                  indexType: indexStream.format.metal,
                                  indexBuffer: indexBuffer,
                                  indexBufferOffset: indexBase + 6,
                                  instanceCount: 1)
    encoder.endEncoding()
    let completed = DispatchSemaphore(value: 0)
    commandBuffer.addCompletedHandler { _ in completed.signal() }
    commandBuffer.commit()
    guard completed.wait(timeout: .now() + .seconds(20)) == .success else {
        throw OracleError("\(definition.id): GPU completion timed out after 20 seconds; submitted work was not cancelled")
    }
    try require(commandBuffer.status == .completed && commandBuffer.error == nil,
                "\(definition.id): Metal execution failed (status \(commandBuffer.status.rawValue)): \(String(describing: commandBuffer.error))")

    var observed = Data(count: depth.width * depth.height)
    observed.withUnsafeMutableBytes { bytes in
        if let destination = bytes.baseAddress {
            stencilLanding.getBytes(destination,
                                    bytesPerRow: depth.width,
                                    from: MTLRegionMake2D(0, 0, depth.width, depth.height),
                                    mipmapLevel: 0)
        }
    }
    return observed
}

/// The stencil-resolve milestone's own fixture, constructed in code.
///
/// This is the one-device check for the question v55 left open
/// (`research/docs/23` §3.3, v59): the reviewed depth pair module draws the
/// v51 edge geometry — a near triangle at z = 0.5 covering NDC x <= 0.25 and
/// a far triangle at z = 0.9 covering everything — three times through a
/// four-sample depth-stencil raster cleared to depth 1.0 and stencil 0. The
/// near triangle increments its covered samples to stencil 1 and the far
/// triangle covers the rest without writing stencil, so the split column's
/// texels carry both a near stencil-1 sample and a far stencil-0 sample. The
/// three passes print one `stencil8` texel per line: the sample0 stencil
/// filter, the depthResolvedSample stencil filter with a min depth resolve,
/// and the depthResolvedSample stencil filter with a max depth resolve. It
/// prints `stencil_resolve_selftest: PASS` only when the stable columns hold
/// — column 0 reads 01 and column 3 reads 00 for all three passes — and the
/// key column distinguishes the depth-resolved sample: column 2 reads 01 for
/// the min resolve and 00 for the max resolve. sample0's column 2 is recorded
/// but not judged, because its landing depends on the rasterizer's sample
/// positions. A device that reduces `depthResolvedSample` to sample0 — or to
/// any one value — fails the key column's split, which is the authoritative
/// answer this check exists to collect.
@available(macOS 11.0, *)
private func stencilResolveSelfTest() throws {
    let fixture = try resolvePairFixture(id: "stencil_resolve_selftest_4x4")
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw OracleError("No default Metal device is available; the stencil-resolve self-test requires an Apple silicon Mac")
    }
    let eligibility = assessDevice(device)
    try require(eligibility.eligible,
                "This oracle requires a named Apple silicon GPU with nonuniform threadgroups and unified memory")
    guard let queue = device.makeCommandQueue() else {
        throw OracleError("Cannot create a Metal command queue")
    }
    diagnostic("native stencil-resolve self-test: device=\(device.name) platform=\(eligibility.platform)")

    let passes: [(name: String, depthFilter: MTLMultisampleDepthResolveFilter,
                  stencilFilter: MTLMultisampleStencilResolveFilter)] = [
        ("sample0", .sample0, .sample0),
        ("depth_resolved_sample_min", .min, .depthResolvedSample),
        ("depth_resolved_sample_max", .max, .depthResolvedSample),
    ]
    var landings = [String: Data]()
    for entry in passes {
        landings[entry.name] = try stencilResolveSelftestPass(
            fixture, device: device, queue: queue,
            depthFilter: entry.depthFilter, stencilFilter: entry.stencilFilter,
            name: entry.name)
    }

    // One byte per `stencil8` texel: the near sample's reviewed value 01 and
    // the far sample's 00, which the texel hexes below compare against.
    let nearByte = Data([0x01])
    let farByte = Data([0x00])
    func texel(_ landing: Data, _ row: Int, _ column: Int) -> Data {
        let offset = row * 4 + column
        return Data(landing[offset..<(offset + 1)])
    }
    // One machine-readable line per texel, row-major, each carrying the three
    // passes' landings for that texel. Column 0 (fully near) and column 3
    // (fully far) are the stable columns, and column 2 is the key column the
    // min and max depth resolves are expected to split.
    var output = ""
    for row in 0..<4 {
        for column in 0..<4 {
            let sample0Hex = hex(texel(landings["sample0"]!, row, column))
            let minHex = hex(texel(landings["depth_resolved_sample_min"]!, row, column))
            let maxHex = hex(texel(landings["depth_resolved_sample_max"]!, row, column))
            output += "stencil_resolve_selftest: row=\(row) column=\(column) "
                + "sample0=\(sample0Hex) depth_resolved_sample(min)=\(minHex) "
                + "depth_resolved_sample(max)=\(maxHex)\n"
        }
    }
    // The PASS judgement is the stable columns for every pass plus the key
    // column's min/max split; sample0's column 2 is deliberately absent.
    var failures = [String]()
    for row in 0..<4 {
        for entry in passes {
            let column0 = texel(landings[entry.name]!, row, 0)
            if column0 != nearByte {
                failures.append("column 0 row \(row) \(entry.name)=\(hex(column0)) "
                                + "expected \(hex(nearByte))")
            }
            let column3 = texel(landings[entry.name]!, row, 3)
            if column3 != farByte {
                failures.append("column 3 row \(row) \(entry.name)=\(hex(column3)) "
                                + "expected \(hex(farByte))")
            }
        }
        let minKey = texel(landings["depth_resolved_sample_min"]!, row, 2)
        if minKey != nearByte {
            failures.append("column 2 row \(row) depth_resolved_sample(min)=\(hex(minKey)) "
                            + "expected \(hex(nearByte))")
        }
        let maxKey = texel(landings["depth_resolved_sample_max"]!, row, 2)
        if maxKey != farByte {
            failures.append("column 2 row \(row) depth_resolved_sample(max)=\(hex(maxKey)) "
                            + "expected \(hex(farByte))")
        }
    }
    if failures.isEmpty {
        output += "stencil_resolve_selftest: PASS\n"
        FileHandle.standardOutput.write(Data(output.utf8))
    } else {
        for failure in failures {
            output += "stencil_resolve_selftest: FAIL (\(failure))\n"
        }
        FileHandle.standardOutput.write(Data(output.utf8))
        // A device that reduces `depthResolvedSample` to one value fails the
        // key column's split, which is the authoritative answer this check
        // exists to collect — so the failure has to propagate rather than
        // print PASS.
        throw OracleError("stencil_resolve_selftest: " + failures.joined(separator: "; "))
    }
}

@available(macOS 11.0, *)
private func assessDevice(_ device: MTLDevice?) -> DeviceProbe {
    let platform = "macOS \(ProcessInfo.processInfo.operatingSystemVersionString)"
    guard let device = device else {
        return DeviceProbe(platform: platform, device: nil, eligible: false,
            reason: "no_default_device", supports_apple4: false, has_unified_memory: false)
    }
    // Apple4 establishes the nonuniform-threadgroup capability. Restricting
    // this initial harness to Apple GPUs also makes shared-memory use explicit.
    let supportsApple4 = device.supportsFamily(.apple4)
    let hasUnifiedMemory = device.hasUnifiedMemory
    let hasName = !device.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    let eligible = hasName && supportsApple4 && hasUnifiedMemory
    return DeviceProbe(platform: platform, device: device.name, eligible: eligible,
        reason: eligible ? "eligible" : "unsupported_features",
        supports_apple4: supportsApple4, has_unified_memory: hasUnifiedMemory)
}

@available(macOS 11.0, *)
private func capture(_ suite: ValidatedSuite) throws -> SuiteResult {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw OracleError("No default Metal device is available; capture requires an Apple silicon Mac")
    }
    let eligibility = assessDevice(device)
    try require(eligibility.eligible,
                "This oracle requires a named Apple silicon GPU with nonuniform threadgroups and unified memory")
    guard let queue = device.makeCommandQueue() else { throw OracleError("Cannot create a Metal command queue") }
    var results = [CaseResult]()
    // loadSuite pins one reviewed source identity per entry. Reuse the pipeline
    // across cases while runCase creates fresh commands and buffers each time.
    var pipelines = [String: MTLComputePipelineState]()
    for fixture in suite.cases {
        // A compute case that carries a heap or indirect section is executable
        // only on the rails its marker names (`research/docs/25` §5.2); this
        // oracle runs no heap placement and no indirect replay, so it omits a
        // case the marker does not name instead of reporting a direct run as
        // if it were that case.
        if let rails = fixture.definition.capture_rails,
           !rails.contains("native-metal") {
            continue
        }
        var selected = [MTLComputePipelineState]()
        for program in fixture.programs {
            let entry = program.definition.entry
            if let cached = pipelines[entry] {
                selected.append(cached)
            } else {
                let library = try device.makeLibrary(source: program.source, options: nil)
                guard let function = library.makeFunction(name: entry) else {
                    throw OracleError("\(fixture.definition.id): Metal function was not found")
                }
                let pipeline = try device.makeComputePipelineState(function: function)
                pipelines[entry] = pipeline
                selected.append(pipeline)
                diagnostic("native pipeline compiled: entry=\(entry)")
            }
        }
        results.append(try runCase(fixture, device: device, queue: queue, pipelines: selected))
    }
    // Render cases run last: they compile the reviewed render module instead of
    // the compute fixtures, and their observable is the attachment's texels.
    // Only the cases this rail's marker names are reported: an unnamed rail
    // that reported a case would present a comparison the suite did not ask
    // for, which `conformance/compare.py` refuses. A device-gated case adds
    // the mask half (`research/docs/23` §3.3, v57d): even a marked case is
    // absent unless this rail's declared mask carries the filter's bit.
    for fixture in suite.renderCases where fixture.definition.capture_rails.contains("native-metal") {
        if let gate = fixture.definition.requires_depth_resolve_filter {
            let bit: UInt64 = gate == "min" ? 2 : 4
            if nativeDepthResolveModes & bit == 0 {
                continue
            }
        }
        if let gate = fixture.definition.requires_stencil_resolve_filter {
            let bit: UInt64 = gate == "depth_resolved_sample" ? 2 : 0
            if nativeStencilResolveModes & bit == 0 {
                continue
            }
        }
        results.append(try runRenderCase(fixture, device: device, queue: queue))
    }
    return SuiteResult(schema_version: 1, suite: suite.name, suite_sha256: suite.sha256,
        backend: "native-metal", allocation_observation: "gpu-buffer-readback",
        depth_resolve_modes: nativeDepthResolveModes,
        stencil_resolve_modes: nativeStencilResolveModes,
        device: device.name, platform: eligibility.platform, results: results)
}

private func writeJSON<T: Encodable>(_ result: T, output: URL? = nil) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    var report = try encoder.encode(result)
    report.append(0x0a)
    if let output = output {
        // Exclusive creation protects against files created during capture.
        try report.write(to: output, options: .withoutOverwriting)
    } else {
        FileHandle.standardOutput.write(report)
    }
}

private func diagnostic(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

do {
    let arguments = Array(CommandLine.arguments.dropFirst())
    if arguments == ["--help"] {
        FileHandle.standardOutput.write(Data((usage + "\n").utf8))
        exit(EXIT_SUCCESS)
    }
    let options = try parseOptions(arguments)
    guard #available(macOS 11.0, *) else { throw OracleError("macOS 11 or later is required") }
    if options.probe {
        // Query capabilities only: no suite, queue, shader, or GPU submission.
        try writeJSON(assessDevice(MTLCreateSystemDefaultDevice()))
        exit(EXIT_SUCCESS)
    }
    if options.renderSelfTest {
        // The one path that reaches the render capture on a device today. The
        // reported bytes are the evidence: four `40 80 c0 ff` texels, never the
        // `fe` clear sentinel the pass started from.
        let result = try renderSelfTest()
        try writeJSON(result)
        exit(EXIT_SUCCESS)
    }
    if options.presentSelfTest {
        // The present equivalent's one-device check: the reported bytes are the
        // evidence, four `40 80 c0 ff` texels, never the `fe` sentinel the
        // target was preset with (`research/docs/24` §6 Step 7).
        let result = try presentSelfTest()
        try writeJSON(result)
        exit(EXIT_SUCCESS)
    }
    if options.vertexSelfTest {
        // The vertex-input milestone's one-device check: the reported bytes are
        // the evidence, four `40 80 c0 ff` texels written through the caller's
        // own vertex stream and index buffer instead of `vertex_id`, never the
        // `fe` clear sentinel (`research/docs/23` §6 Step 3.3).
        let result = try vertexSelfTest()
        try writeJSON(result)
        exit(EXIT_SUCCESS)
    }
    if options.mrtSelfTest {
        // The MRT milestone's one-device check: the reported bytes are the
        // evidence, location 0's four `40 80 c0 ff` texels and location 1's
        // four `ff 80 40 c0` texels, never the `fe` clear sentinel
        // (`conformance/RENDER-CAPTURE.md` §10).
        let result = try mrtSelfTest()
        try writeJSON(result)
        exit(EXIT_SUCCESS)
    }
    if options.heapSelfTest {
        // The heap milestone's one-device check: two buffers in one MTLHeap,
        // the reviewed copy_word kernel across them, and a write-buffer
        // readback that must be the reviewed word rather than the sentinel
        // (`research/docs/25` §6 Step 7a).
        let result = try heapSelfTest()
        try writeJSON(result)
        exit(EXIT_SUCCESS)
    }
    if options.depthResolveSelfTest {
        // The depth-resolve milestone's one-device check: the reviewed depth
        // pair draws the v51 edge geometry once per filter, and the per-texel
        // landings this check prints are the authoritative answer for whether
        // the Apple device executes Min/Max. It exits nonzero unless the
        // stable columns are as reviewed and the key column distinguishes min
        // from max (`research/docs/23` §3.3, v57e).
        try depthResolveSelfTest()
        exit(EXIT_SUCCESS)
    }
    if options.stencilResolveSelfTest {
        // The stencil-resolve milestone's one-device check: the reviewed pair
        // draws the v51 edge geometry once per filter, and the per-texel
        // stencil8 landings this check prints are the authoritative answer for
        // whether the Apple device executes `depthResolvedSample` as the
        // sample the depth resolve filter selects. It exits nonzero unless the
        // stable columns are as reviewed and the key column distinguishes the
        // min-resolved from the max-resolved stencil (`research/docs/23` §3.3,
        // v59).
        try stencilResolveSelfTest()
        exit(EXIT_SUCCESS)
    }
    guard let suiteURL = options.suite else { throw OracleError("--suite is required") }
    let suite = try loadSuite(suiteURL)
    if options.validateOnly {
        diagnostic("Validated \(suite.name): \(suite.cases.count) cases, suite SHA-256 \(suite.sha256); no GPU work submitted")
    } else {
        let result = try capture(suite)
        try writeJSON(result, output: options.output)
    }
} catch {
    diagnostic("native-metal-oracle: \(error)")
    exit(EXIT_FAILURE)
}
