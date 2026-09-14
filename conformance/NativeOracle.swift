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
    let attachment: RenderAttachmentDefinition
    let expected_hex: String
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
    let attachment: ValidatedRenderAttachment
}

private struct ValidatedRenderAttachment {
    let allocation: UInt64
    let view: UInt64
    let width: Int
    let height: Int
    let load: String
    let clearComponents: [Double]
    let initial: Data?
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

private struct SuiteResult: Encodable {
    let schema_version: UInt64
    let suite: String
    let suite_sha256: String
    let backend: String
    let allocation_observation: String
    let device: String
    let platform: String
    let results: [CaseResult]
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
}

private let usage = """
Usage: native-metal-oracle --suite PATH [--output PATH]
       native-metal-oracle --suite PATH --validate-suite
       native-metal-oracle --probe
       native-metal-oracle --render-selftest
       native-metal-oracle --present-selftest
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
The 20-second completion timeout does not cancel submitted GPU work.
"""

private func parseOptions(_ arguments: [String]) throws -> Options {
    var suite: URL?
    var output: URL?
    var validateOnly = false
    var probe = false
    var renderSelfTest = false
    var presentSelfTest = false
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
        default:
            throw OracleError("Unknown argument: \(argument)\n\(usage)")
        }
    }
    if probe {
        try require(suite == nil && output == nil && !validateOnly && !renderSelfTest && !presentSelfTest,
                    "--probe cannot be combined with --suite, --output, --validate-suite, --render-selftest, or --present-selftest")
        return Options(suite: nil, output: nil, validateOnly: false, probe: true,
                       renderSelfTest: false, presentSelfTest: false)
    }
    if renderSelfTest {
        try require(suite == nil && output == nil && !validateOnly && !presentSelfTest,
                    "--render-selftest cannot be combined with --suite, --output, --validate-suite, or --present-selftest")
        return Options(suite: nil, output: nil, validateOnly: false, probe: false,
                       renderSelfTest: true, presentSelfTest: false)
    }
    if presentSelfTest {
        try require(suite == nil && output == nil && !validateOnly,
                    "--present-selftest cannot be combined with --suite, --output, or --validate-suite")
        return Options(suite: nil, output: nil, validateOnly: false, probe: false,
                       renderSelfTest: false, presentSelfTest: true)
    }
    try require(suite != nil, "--suite is required\n\(usage)")
    try require(!validateOnly || output == nil, "--output cannot be used with --validate-suite")
    if let outputURL = output {
        try require(!FileManager.default.fileExists(atPath: outputURL.path),
                    "Output already exists: \(outputURL.path)")
    }
    return Options(suite: suite, output: output, validateOnly: validateOnly, probe: false,
                   renderSelfTest: false, presentSelfTest: false)
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
    case "copy_word", "copy_seed_a", "copy_seed_b", "copy_pingpong",
         "alias_disjoint_pair", "alias_disjoint_pair_reversed":
        try require(definition.entry == "copy_word"
                    && definition.grid == [1, 1, 1] && definition.local == [1, 1, 1],
                    "copy_word: unsupported entry or dispatch shape")
        try require(definition.buffers.count == 2, "copy_word: expected two buffers")
        try require(definition.buffers.contains { $0.binding == 0 && $0.access == "read" && $0.length == 4 }
                    && definition.buffers.contains { $0.binding == 1 && $0.access == "write" && $0.length == 4 },
                    "copy_word: expected a 4-byte read buffer at 0 and write buffer at 1")
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
        // v13's declaring case is the one exception: its read view *is* the
        // whole attachment allocation (`offset == 0`,
        // `allocation_size == length`), because the render case stores into
        // exactly those bytes and the comparison is against that view. There is
        // no neighbouring byte to guard with, and the render path's own
        // sentinel-versus-fragment check is what keeps an extent mistake
        // observable there (`conformance/RENDER-CAPTURE.md` §3). The exception
        // is written down here rather than loosening the rule for every case.
        let wholeAllocationDeclaringView =
            declaringShapeIDs.contains(definition.id)
            && buffer.access == "read" && buffer.offset == 0
            && buffer.allocation_size == buffer.length
        let end = buffer.offset + buffer.length
        try require(wholeAllocationDeclaringView
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
    default:
        throw OracleError("Only compute-buffer-v1 through compute-buffer-v15 are supported")
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
}

/// The reviewed render fixture.
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
                               sha256: "7430cd19a3497582618226066e95fb6f4ead9071f83b00c53398ccab8ba9d7de"))
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
    var cases = [ValidatedRender]()
    for definition in suite.render_cases ?? [] {
        cases.append(try validateRenderCase(definition, root: root))
    }
    let ids = cases.map { $0.definition.id }
    try require(Set(ids).count == ids.count, "\(suite.suite): duplicate render case id")
    let computeIDs = Set(suite.cases.map { $0.id })
    for id in ids {
        try require(!computeIDs.contains(id),
                    "\(suite.suite): render case \(id) repeats a compute case id")
    }
    return cases
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
    let reviewed = reviewedRenderModule()
    try require(definition.vertex_entry == reviewed.vertex_entry
                && definition.fragment_entry == reviewed.fragment_entry
                && definition.metal == reviewed.metal,
                "\(definition.id): unreviewed render pipeline identity")
    let sourceBytes = try loadRenderSource(reviewed.metal, root: root)
    guard let source = String(data: sourceBytes, encoding: .utf8) else {
        throw OracleError("\(definition.id): reviewed MSL source is not UTF-8")
    }
    let attachment = definition.attachment
    try require(attachment.format == "rgba8_unorm",
                "\(definition.id): unsupported attachment format")
    try require(attachment.width == 2 && attachment.height == 2,
                "\(definition.id): the first render increment renders into a 2x2 attachment")
    try require(attachment.allocation > 0 && attachment.view > 0,
                "\(definition.id): zero attachment identity")
    try require(attachment.store == "store",
                "\(definition.id): the attachment has to be stored for a byte comparison")
    try require(definition.vertices == 3,
                "\(definition.id): expected the full-screen triangle")
    try require(definition.viewport == [0, 0, UInt64(attachment.width), UInt64(attachment.height)],
                "\(definition.id): the viewport must cover the attachment")
    let byteCount = attachment.width * attachment.height * 4
    let expected = try decodeHex(definition.expected_hex, context: "\(definition.id) expected texels")
    try require(expected.count == byteCount,
                "\(definition.id): expected texel bytes do not match the attachment")
    // Full coverage is the milestone's whole point (`research/docs/23` §1.3): the
    // expectation admits only identical texels, so a partially covered
    // attachment cannot be asserted as correct.
    let texel = Data(expected.prefix(4))
    var texels = 0
    for offset in stride(from: 0, to: expected.count, by: 4) {
        try require(Data(expected[offset..<(offset + 4)]) == texel,
                    "\(definition.id): the milestone expects every texel to equal the fragment output")
        texels += 1
    }
    try require(texels == attachment.width * attachment.height,
                "\(definition.id): attachment texel count mismatch")
    let clearComponents: [Double]
    let initial: Data?
    switch attachment.load {
    case "clear":
        guard let clearHex = attachment.clear_hex else {
            throw OracleError("\(definition.id): a clear attachment needs clear_hex")
        }
        let clearBytes = try decodeHex(clearHex, context: "\(definition.id) clear colour")
        try require(clearBytes.count == 4, "\(definition.id): a clear colour is four bytes")
        // The sentinel has to be distinguishable from the fragment output, or a
        // pass that never ran would satisfy the expectation.
        try require(clearBytes != texel,
                    "\(definition.id): the clear colour equals the expected texel")
        try require(attachment.initial_hex == nil,
                    "\(definition.id): a cleared attachment carries no initial bytes")
        clearComponents = [Double(clearBytes[0]) / 255.0, Double(clearBytes[1]) / 255.0,
                           Double(clearBytes[2]) / 255.0, Double(clearBytes[3]) / 255.0]
        initial = nil
    case "load":
        guard let initialHex = attachment.initial_hex else {
            throw OracleError("\(definition.id): a loaded attachment needs its previous texels")
        }
        let previous = try decodeHex(initialHex, context: "\(definition.id) initial texels")
        try require(previous.count == byteCount,
                    "\(definition.id): initial texels do not match the attachment")
        try require(previous != expected,
                    "\(definition.id): the initial texels equal the expectation")
        clearComponents = []
        initial = previous
    default:
        throw OracleError("\(definition.id): unsupported attachment load op \(attachment.load)")
    }
    return ValidatedRender(definition: definition, source: source,
                           attachment: ValidatedRenderAttachment(
                               allocation: attachment.allocation, view: attachment.view,
                               width: attachment.width, height: attachment.height,
                               load: attachment.load, clearComponents: clearComponents,
                               initial: initial, expected: expected))
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

/// One offscreen render case: a 2x2 `rgba8Unorm` attachment, the reviewed
/// two-entry pipeline and a full-screen-triangle draw, read back as texels.
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
    let attachment = fixture.attachment
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm,
        width: attachment.width,
        height: attachment.height,
        mipmapped: false)
    // The attachment is a render target, not a sampled source. Shared storage is
    // what makes its texels CPU-visible for the readback on the unified-memory
    // device this oracle requires, the same reason the sampled texture rail uses
    // it (`research/docs/16` §4.8).
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
    // Attachment 0 is the only colour attachment the first increment admits, and
    // its pixel format is the one the reviewed fragment was written for.
    pipelineDescriptor.colorAttachments[0].pixelFormat = .rgba8Unorm
    let pipeline = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)

    let pass = MTLRenderPassDescriptor()
    // `colorAttachments[0]` is an implicitly unwrapped optional on the Swift
    // side of Metal, but referencing a member before unwrapping is a compile
    // error under `-warnings-as-errors`; unwrap it once, explicitly.
    guard let color = pass.colorAttachments[0] else {
        throw OracleError("\(definition.id): cannot reach the colour attachment descriptor")
    }
    color.texture = target
    color.storeAction = .store
    if attachment.load == "clear" {
        color.loadAction = .clear
        guard attachment.clearComponents.count == 4 else {
            throw OracleError("\(definition.id): a clear colour is four components")
        }
        color.clearColor = MTLClearColor(red: attachment.clearComponents[0],
                                         green: attachment.clearComponents[1],
                                         blue: attachment.clearComponents[2],
                                         alpha: attachment.clearComponents[3])
    } else {
        color.loadAction = .load
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
    // The viewport is explicit because the contract carries it, even though the
    // first increment only accepts the attachment-covering default.
    encoder.setViewport(MTLViewport(originX: 0, originY: 0,
                                    width: Double(attachment.width),
                                    height: Double(attachment.height),
                                    znear: 0, zfar: 1))
    encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: Int(definition.vertices))
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

    var observed = Data(count: attachment.width * attachment.height * 4)
    observed.withUnsafeMutableBytes { bytes in
        if let destination = bytes.baseAddress {
            target.getBytes(destination,
                            bytesPerRow: attachment.width * 4,
                            from: MTLRegionMake2D(0, 0, attachment.width, attachment.height),
                            mipmapLevel: 0)
        }
    }
    try require(observed == attachment.expected,
                "\(definition.id): attachment bytes \(hex(observed)) do not match the reviewed expectation \(hex(attachment.expected))")
    // One allocation, one writeback: the attachment's own texels.
    return CaseResult(id: definition.id, completion: "CompletedVisible",
                      writebacks: [Writeback(allocation: attachment.allocation, view: attachment.view,
                                             offset: 0, bytes_hex: hex(observed))],
                      allocations: [AllocationResult(allocation: attachment.allocation,
                                                     bytes_hex: hex(observed))])
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
        attachment: RenderAttachmentDefinition(
            allocation: 900, view: 910, format: "rgba8_unorm",
            width: 2, height: 2, load: "clear", store: "store",
            clear_hex: "fefefefe", initial_hex: nil),
        expected_hex: "4080c0ff4080c0ff4080c0ff4080c0ff",
        // The self-test runs on this rail by construction; the marker is the
        // same one suite-v13 names for it.
        capture_rails: ["native-metal"])
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
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
        attachment: RenderAttachmentDefinition(
            allocation: 900, view: 910, format: "rgba8_unorm",
            width: 2, height: 2, load: "load", store: "store",
            clear_hex: nil, initial_hex: hex(sentinel)),
        expected_hex: "4080c0ff4080c0ff4080c0ff4080c0ff",
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
    // for, which `conformance/compare.py` refuses.
    for fixture in suite.renderCases where fixture.definition.capture_rails.contains("native-metal") {
        results.append(try runRenderCase(fixture, device: device, queue: queue))
    }
    return SuiteResult(schema_version: 1, suite: suite.name, suite_sha256: suite.sha256,
        backend: "native-metal", allocation_observation: "gpu-buffer-readback",
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
