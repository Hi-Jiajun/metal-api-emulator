// Capture native Metal observations for the bounded compute-buffer-v1 suite.
// Build on macOS with Swift 5 language mode and link Foundation, Metal,
// CoreGraphics, and CryptoKit. This file does not implement ComputeProvider.
import Foundation
import Metal
import CoreGraphics
import CryptoKit
import Dispatch
import Darwin

private let maximumFileBytes = 1_048_576
private let maximumAllocationBytes: UInt64 = 1_048_576

private struct OracleError: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}

private func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw OracleError(message) }
}

private struct SourceDefinition: Decodable {
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

private struct CaseDefinition: Decodable {
    let id: String
    let entry: String
    let grid: [UInt64]
    let local: [UInt64]
    let air: SourceDefinition
    let metal: SourceDefinition
    let buffers: [BufferDefinition]
    let expected_writebacks: [Writeback]
}

private struct SuiteDefinition: Decodable {
    let schema_version: UInt64
    let suite: String
    let guard_byte: UInt8
    let cases: [CaseDefinition]
}

private struct ValidatedBuffer {
    let definition: BufferDefinition
    let backing: Data
}

private struct ValidatedCase {
    let definition: CaseDefinition
    let metalSource: String
    let buffers: [ValidatedBuffer]
}

private struct ValidatedSuite {
    let name: String
    let sha256: String
    let cases: [ValidatedCase]
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

private struct Options {
    let suite: URL
    let output: URL?
    let validateOnly: Bool
}

private let usage = """
Usage: native-metal-oracle --suite PATH [--output PATH]
       native-metal-oracle --suite PATH --validate-suite
       native-metal-oracle --help

Capture the two supported cases using native Metal on Apple silicon macOS 11+.
Without --output, the successful JSON report goes to stdout. Existing output
files are never overwritten. Diagnostics go to stderr. --validate-suite checks
the fixture and both shader source hashes without creating a Metal device.
The 20-second completion timeout does not cancel submitted GPU work.
"""

private func parseOptions(_ arguments: [String]) throws -> Options {
    var suite: URL?
    var output: URL?
    var validateOnly = false
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
        default:
            throw OracleError("Unknown argument: \(argument)\n\(usage)")
        }
    }
    guard let suiteURL = suite else { throw OracleError("--suite is required\n\(usage)") }
    try require(!validateOnly || output == nil, "--output cannot be used with --validate-suite")
    if let outputURL = output {
        try require(!FileManager.default.fileExists(atPath: outputURL.path),
                    "Output already exists: \(outputURL.path)")
    }
    return Options(suite: suiteURL, output: output, validateOnly: validateOnly)
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

private func validateShape(_ definition: CaseDefinition) throws {
    try require(definition.grid.count == 3 && definition.local.count == 3,
                "\(definition.id): grid and local need three dimensions")
    try require(definition.grid.allSatisfy { $0 > 0 && $0 <= 1024 }
                && definition.local.allSatisfy { $0 > 0 && $0 <= 1024 },
                "\(definition.id): dimensions must be in 1...1024")
    try require(definition.local.reduce(UInt64(1), *) <= 1024,
                "\(definition.id): excessive threads per threadgroup")
    switch definition.id {
    case "copy_word":
        try require(definition.entry == "copy_word"
                    && definition.grid == [1, 1, 1] && definition.local == [1, 1, 1],
                    "copy_word: unsupported entry or dispatch shape")
        try require(definition.buffers.count == 2, "copy_word: expected two buffers")
        try require(definition.buffers.contains { $0.binding == 0 && $0.access == "read" && $0.length == 4 }
                    && definition.buffers.contains { $0.binding == 1 && $0.access == "write" && $0.length == 4 },
                    "copy_word: expected a 4-byte read buffer at 0 and write buffer at 1")
    case "indexed_boundary":
        try require(definition.entry == "kernel_dispatch_threads_boundary_barrier"
                    && definition.grid == [10, 3, 1] && definition.local == [8, 2, 1],
                    "indexed_boundary: unsupported entry or dispatch shape")
        try require(definition.buffers.count == 1, "indexed_boundary: expected one buffer")
        let buffer = definition.buffers[0]
        try require(buffer.binding == 0 && buffer.access == "write" && buffer.length == 120,
                    "indexed_boundary: expected a 120-byte write buffer at 0")
    default:
        throw OracleError("Unsupported case: \(definition.id)")
    }
}

private func validateBuffers(_ definition: CaseDefinition, guardByte: UInt8) throws -> [ValidatedBuffer] {
    var bindings = Set<UInt64>()
    var allocations = Set<UInt64>()
    var views = Set<UInt64>()
    var buffers = [ValidatedBuffer]()
    for buffer in definition.buffers {
        let context = "\(definition.id) binding \(buffer.binding)"
        try require(bindings.insert(buffer.binding).inserted, "\(context): duplicate binding")
        try require(allocations.insert(buffer.allocation).inserted, "\(context): duplicate allocation")
        try require(views.insert(buffer.view).inserted, "\(context): duplicate view")
        try require(buffer.allocation > 0 && buffer.view > 0, "\(context): zero resource identity")
        try require(buffer.access == "read" || buffer.access == "write", "\(context): unsupported access")
        try require(buffer.length > 0 && buffer.allocation_size <= maximumAllocationBytes,
                    "\(context): allocation must be nonempty and at most 1 MiB")
        try require(buffer.offset <= buffer.allocation_size
                    && buffer.length <= buffer.allocation_size - buffer.offset,
                    "\(context): view extends beyond allocation")
        try require(buffer.offset % 4 == 0, "\(context): uint binding offset needs 4-byte alignment")
        // Each owned view must have a canary prefix and suffix to make an
        // offset/extent mismatch observable. Bounds above make addition safe.
        let end = buffer.offset + buffer.length
        try require(buffer.offset >= 4 && buffer.allocation_size - end >= 4,
                    "\(context): expected at least four guard bytes before and after the view")
        let initial = try decodeHex(buffer.initial_hex, context: context)
        try require(UInt64(initial.count) == buffer.length, "\(context): initial data length mismatch")
        var backing = Data(repeating: guardByte, count: Int(buffer.allocation_size))
        backing.replaceSubrange(Int(buffer.offset)..<Int(end), with: initial)
        buffers.append(ValidatedBuffer(definition: buffer, backing: backing))
    }

    let writable = definition.buffers.filter { $0.access == "write" }
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
    try require(suite.schema_version == 1 && suite.suite == "compute-buffer-v1",
                "Only compute-buffer-v1 schema version 1 is supported")
    try require(suite.cases.count == 2 && Set(suite.cases.map { $0.id }) == Set(["copy_word", "indexed_boundary"]),
                "The suite must contain exactly copy_word and indexed_boundary")
    let root = url.deletingLastPathComponent()
    var cases = [ValidatedCase]()
    for definition in suite.cases {
        try validateShape(definition)
        let buffers = try validateBuffers(definition, guardByte: suite.guard_byte)
        let metalBytes: Data
        if definition.id == "copy_word" {
            _ = try validateSource(definition.air, root: root,
                path: "../examples/metal-smoke/shaders/kernel_copy_word.ll",
                digest: "292c3e1ff300fd08bf5e39aaa9abe352842eced807138f863e05056f39c56d99")
            metalBytes = try validateSource(definition.metal, root: root,
                path: "shaders/copy_word.metal",
                digest: "7bfa419aef6eb0abcbec045c1bc15651b2d8f0a7591e07448edc6de6522141bc")
        } else {
            _ = try validateSource(definition.air, root: root,
                path: "../examples/metal-smoke/shaders/kernel_dispatch_threads_boundary_barrier.ll",
                digest: "95076cf4199734f848fd6d761dce13addc7b55354b4d8ee2be16e59287ea5945")
            metalBytes = try validateSource(definition.metal, root: root,
                path: "shaders/indexed_boundary.metal",
                digest: "7684e493a8704127e39dace5476a006fac564224909c667a57fb5ac9d8291b06")
        }
        guard let metalSource = String(data: metalBytes, encoding: .utf8) else {
            throw OracleError("\(definition.id): MSL source is not UTF-8")
        }
        cases.append(ValidatedCase(definition: definition, metalSource: metalSource, buffers: buffers))
    }
    return ValidatedSuite(name: suite.suite, sha256: sha256(raw), cases: cases)
}

private func metalSize(_ dimensions: [UInt64]) -> MTLSize {
    MTLSize(width: Int(dimensions[0]), height: Int(dimensions[1]), depth: Int(dimensions[2]))
}

@available(macOS 11.0, *)
private func runCase(_ fixture: ValidatedCase, device: MTLDevice, queue: MTLCommandQueue) throws -> CaseResult {
    let definition = fixture.definition
    let library = try device.makeLibrary(source: fixture.metalSource, options: nil)
    guard let function = library.makeFunction(name: definition.entry) else {
        throw OracleError("\(definition.id): Metal function was not found")
    }
    let pipeline = try device.makeComputePipelineState(function: function)
    let local = metalSize(definition.local)
    let limits = device.maxThreadsPerThreadgroup
    try require(local.width <= limits.width && local.height <= limits.height && local.depth <= limits.depth,
                "\(definition.id): local dimensions exceed device limits")
    try require(local.width * local.height * local.depth <= pipeline.maxTotalThreadsPerThreadgroup,
                "\(definition.id): local thread count exceeds pipeline limit")

    // makeCommandBuffer() retains referenced resources until GPU completion.
    // In particular, a CPU timeout below must not release submitted buffers.
    guard let commandBuffer = queue.makeCommandBuffer() else {
        throw OracleError("\(definition.id): cannot create a command buffer")
    }
    try require(commandBuffer.retainedReferences, "\(definition.id): command buffer does not retain resources")
    commandBuffer.label = "native oracle: \(definition.id)"
    var resources = [MTLBuffer]()
    for buffer in fixture.buffers {
        try require(buffer.backing.count <= device.maxBufferLength,
                    "\(definition.id): allocation exceeds the Metal buffer limit")
        guard let resource = device.makeBuffer(length: buffer.backing.count, options: .storageModeShared) else {
            throw OracleError("\(definition.id): cannot allocate a shared Metal buffer")
        }
        try require(resource.storageMode == .shared, "\(definition.id): shared storage was not selected")
        buffer.backing.withUnsafeBytes { bytes in
            if let source = bytes.baseAddress {
                resource.contents().copyMemory(from: source, byteCount: bytes.count)
            }
        }
        resources.append(resource)
    }
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        throw OracleError("\(definition.id): cannot create a compute encoder")
    }
    encoder.setComputePipelineState(pipeline)
    for (buffer, resource) in zip(fixture.buffers, resources) {
        encoder.setBuffer(resource, offset: Int(buffer.definition.offset), index: Int(buffer.definition.binding))
    }
    encoder.dispatchThreads(metalSize(definition.grid), threadsPerThreadgroup: local)
    encoder.endEncoding()

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

    var allocations = [AllocationResult]()
    var writebacks = [Writeback]()
    for (buffer, resource) in zip(fixture.buffers, resources) {
        let specification = buffer.definition
        // Only completed shared resources are CPU-visible. Copy the complete
        // backing allocation so the comparator can independently inspect it.
        let observed = Data(bytes: resource.contents(), count: buffer.backing.count)
        let start = Int(specification.offset)
        let end = start + Int(specification.length)
        if specification.access == "read" {
            try require(observed == buffer.backing, "\(definition.id): read-only allocation \(specification.allocation) changed")
        } else {
            try require(observed.prefix(start) == buffer.backing.prefix(start)
                        && observed.suffix(observed.count - end) == buffer.backing.suffix(buffer.backing.count - end),
                        "\(definition.id): guard bytes changed in allocation \(specification.allocation)")
            writebacks.append(Writeback(allocation: specification.allocation, view: specification.view,
                offset: specification.offset, bytes_hex: hex(observed.subdata(in: start..<end))))
        }
        allocations.append(AllocationResult(allocation: specification.allocation, bytes_hex: hex(observed)))
    }
    return CaseResult(id: definition.id, completion: "CompletedVisible", writebacks: writebacks, allocations: allocations)
}

@available(macOS 11.0, *)
private func capture(_ suite: ValidatedSuite) throws -> SuiteResult {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw OracleError("No default Metal device is available; capture requires an Apple silicon Mac")
    }
    // Apple4 establishes the nonuniform-threadgroup capability. Restricting
    // this initial harness to Apple GPUs also makes shared-memory use explicit.
    try require(device.supportsFamily(.apple4) && device.hasUnifiedMemory,
                "This oracle requires an Apple silicon GPU with nonuniform threadgroups and unified memory")
    try require(!device.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                "Metal device name is empty")
    guard let queue = device.makeCommandQueue() else { throw OracleError("Cannot create a Metal command queue") }
    var results = [CaseResult]()
    for fixture in suite.cases {
        results.append(try runCase(fixture, device: device, queue: queue))
    }
    let platform = "macOS \(ProcessInfo.processInfo.operatingSystemVersionString)"
    return SuiteResult(schema_version: 1, suite: suite.name, suite_sha256: suite.sha256,
        backend: "native-metal", allocation_observation: "gpu-buffer-readback",
        device: device.name, platform: platform, results: results)
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
    let suite = try loadSuite(options.suite)
    if options.validateOnly {
        diagnostic("Validated \(suite.name): \(suite.cases.count) cases, suite SHA-256 \(suite.sha256); no GPU work submitted")
    } else {
        let result = try capture(suite)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var report = try encoder.encode(result)
        report.append(0x0a)
        if let output = options.output {
            // Exclusive creation protects against files created during capture.
            try report.write(to: output, options: .withoutOverwriting)
        } else {
            FileHandle.standardOutput.write(report)
        }
    }
} catch {
    diagnostic("native-metal-oracle: \(error)")
    exit(EXIT_FAILURE)
}
