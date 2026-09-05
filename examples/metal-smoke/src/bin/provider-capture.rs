//! Capture a Vulkan provider run of the shared, versioned native-oracle suite.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, BufferAccess, BufferSource, BufferView, CompletionDisposition,
    CompletionPolicy, ComputePass, ComputeProvider, ComputeTrace, Dispatch, DispatchKind,
    DispatchType, OperationId, ResourceTableSnapshot, SemanticDigest, ViewId,
    PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::Device;
use metal_api_vulkan::{VulkanComputeProvider, VulkanExecutor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const MAX_BYTES: usize = 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Suite {
    schema_version: u32,
    suite: String,
    guard_byte: u8,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    entry: String,
    grid: [u64; 3],
    local: [u64; 3],
    air: Source,
    metal: Source,
    buffers: Vec<Buffer>,
    expected_writebacks: Vec<Writeback>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    path: String,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Buffer {
    binding: u32,
    allocation: u64,
    view: u64,
    offset: u64,
    length: u64,
    allocation_size: u64,
    access: String,
    initial_hex: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Writeback {
    allocation: u64,
    view: u64,
    offset: u64,
    bytes_hex: String,
}

#[derive(Serialize)]
struct Allocation {
    allocation: u64,
    bytes_hex: String,
}

#[derive(Serialize)]
struct CaseResult {
    id: String,
    completion: &'static str,
    writebacks: Vec<Writeback>,
    allocations: Vec<Allocation>,
}

#[derive(Serialize)]
struct Capture {
    schema_version: u32,
    suite: String,
    suite_sha256: String,
    backend: &'static str,
    allocation_observation: &'static str,
    device: String,
    platform: String,
    results: Vec<CaseResult>,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut suite_path = None;
    let mut output_path = None;
    while let Some(flag) = args.next() {
        if flag == "--help" {
            println!(
                "usage: provider-capture --suite conformance/suite.json [--output capture.json]"
            );
            return Ok(());
        }
        let destination = if flag == "--suite" && suite_path.is_none() {
            &mut suite_path
        } else if flag == "--output" && output_path.is_none() {
            &mut output_path
        } else {
            return Err("unknown or duplicate argument; use --help".into());
        };
        *destination = Some(PathBuf::from(args.next().ok_or("missing argument value")?));
    }
    let suite_path = suite_path.ok_or("--suite is required")?;
    if output_path.as_ref().is_some_and(|path| path.exists()) {
        return Err("refusing to overwrite an existing capture".into());
    }
    // Validate every source and case before creating the Vulkan device.
    let raw = read_bounded(&suite_path, 65536)?;
    let suite: Suite = serde_json::from_slice(&raw)?;
    validate_suite(&suite)?;
    let directory = suite_path.parent().unwrap_or(Path::new("."));
    let mut sources = Vec::new();
    for case in &suite.cases {
        let air = verified_source(directory, &case.air)?;
        let _metal = verified_source(directory, &case.metal)?;
        sources.push(String::from_utf8(air)?);
    }
    let identity = hex(&Sha256::digest(&raw));
    let executor = VulkanExecutor::new()?;
    let provider = VulkanComputeProvider::with_executor(Arc::clone(&executor))
        .map_err(|error| format!("create provider: {error:?}"))?;
    let device = Device::new(executor);
    let mut results = Vec::new();
    for (index, (case, air)) in suite.cases.iter().zip(sources).enumerate() {
        results.push(run_case(
            &provider,
            &device,
            case,
            air,
            index as u64 + 1,
            suite.guard_byte,
            &identity,
        )?);
    }
    let capture = Capture {
        schema_version: 1,
        suite: suite.suite,
        suite_sha256: identity,
        backend: "vulkan",
        allocation_observation: "host-writeback-landing",
        device: provider.device_name().to_owned(),
        platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        results,
    };
    let mut bytes = serde_json::to_vec_pretty(&capture)?;
    bytes.push(b'\n');
    if let Some(path) = output_path {
        // create_new also closes the race after the initial existence check.
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?
            .write_all(&bytes)?;
    } else {
        std::io::stdout().lock().write_all(&bytes)?;
    }
    Ok(())
}

fn validate_suite(suite: &Suite) -> Result<()> {
    if suite.schema_version != 1 || suite.suite != "compute-buffer-v1" || suite.cases.len() != 2 {
        return Err("unsupported suite identity/version/case count".into());
    }
    let mut ids = BTreeSet::new();
    for case in &suite.cases {
        let (air_path, air_hash, metal_path, metal_hash) = match case.id.as_str() {
            "copy_word" => (
                "../examples/metal-smoke/shaders/kernel_copy_word.ll",
                "292c3e1ff300fd08bf5e39aaa9abe352842eced807138f863e05056f39c56d99",
                "shaders/copy_word.metal",
                "7bfa419aef6eb0abcbec045c1bc15651b2d8f0a7591e07448edc6de6522141bc",
            ),
            "indexed_boundary" => (
                "../examples/metal-smoke/shaders/kernel_dispatch_threads_boundary_barrier.ll",
                "95076cf4199734f848fd6d761dce13addc7b55354b4d8ee2be16e59287ea5945",
                "shaders/indexed_boundary.metal",
                "7684e493a8704127e39dace5476a006fac564224909c667a57fb5ac9d8291b06",
            ),
            _ => return Err("unknown case identity".into()),
        };
        if case.air.path != air_path
            || case.air.sha256 != air_hash
            || case.metal.path != metal_path
            || case.metal.sha256 != metal_hash
        {
            return Err("unreviewed shader identity".into());
        }
        if !ids.insert(&case.id) {
            return Err("duplicate case identity".into());
        }
        let (entry, grid, local, accesses, lengths): (_, _, _, &[&str], &[u64]) =
            match case.id.as_str() {
                "copy_word" => (
                    "copy_word",
                    [1, 1, 1],
                    [1, 1, 1],
                    &["read", "write"],
                    &[4, 4],
                ),
                "indexed_boundary" => (
                    "kernel_dispatch_threads_boundary_barrier",
                    [10, 3, 1],
                    [8, 2, 1],
                    &["write"],
                    &[120],
                ),
                _ => return Err("unknown case identity".into()),
            };
        if case.entry != entry
            || case.grid != grid
            || case.local != local
            || case.buffers.len() != accesses.len()
        {
            return Err(
                format!("case {} is outside the qualified dispatch subset", case.id).into(),
            );
        }
        let mut allocations = BTreeSet::new();
        let mut views = BTreeSet::new();
        for (index, buffer) in case.buffers.iter().enumerate() {
            let end = buffer
                .offset
                .checked_add(buffer.length)
                .ok_or("view range overflow")?;
            if buffer.binding != index as u32
                || buffer.access != accesses[index]
                || buffer.length != lengths[index]
                || buffer.allocation_size > MAX_BYTES as u64
                || end > buffer.allocation_size
                || buffer.offset < 4
                || buffer.allocation_size - end < 4
                || !buffer.offset.is_multiple_of(4)
                || buffer.allocation == 0
                || buffer.view == 0
                || !allocations.insert(buffer.allocation)
                || !views.insert(buffer.view)
            {
                return Err(format!("invalid buffer declaration in {}", case.id).into());
            }
            if unhex(&buffer.initial_hex)?.len() as u64 != buffer.length {
                return Err("initial data length differs from declared view length".into());
            }
        }
        let writable: Vec<_> = case
            .buffers
            .iter()
            .filter(|b| b.access == "write")
            .collect();
        if case.expected_writebacks.len() != writable.len() {
            return Err("expected result does not cover writable views".into());
        }
        for (expected, buffer) in case.expected_writebacks.iter().zip(writable) {
            if expected.allocation != buffer.allocation
                || expected.view != buffer.view
                || expected.offset != buffer.offset
                || unhex(&expected.bytes_hex)?.len() as u64 != buffer.length
            {
                return Err("expected result identity/range mismatch".into());
            }
        }
    }
    Ok(())
}

fn run_case(
    provider: &VulkanComputeProvider,
    device: &Device,
    case: &Case,
    air: String,
    operation: u64,
    guard: u8,
    suite_digest: &str,
) -> Result<CaseResult> {
    let function = device.new_library_with_air(air)?.function(&case.entry)?;
    let pipeline = provider
        .compile_pipeline(
            &function,
            SemanticDigest::new(
                "suite-sha256-case-v1",
                format!("{suite_digest}:{}", case.id).into_bytes(),
            )?,
        )
        .map_err(|error| format!("compile {}: {error:?}", case.id))?;
    let mut resources = ResourceTableSnapshot::new();
    let mut views = Vec::new();
    let mut allocations = Vec::new();
    for buffer in &case.buffers {
        let access = match buffer.access.as_str() {
            "read" => BufferAccess::Read,
            "write" => BufferAccess::Write,
            _ => return Err("unsupported access".into()),
        };
        let reflected = pipeline
            .contract
            .buffer_bindings
            .iter()
            .find(|b| b.metal_binding == buffer.binding)
            .ok_or("source reflection omitted fixture buffer")?;
        if reflected.access != access {
            return Err("source/fixture access mismatch".into());
        }
        let initial = unhex(&buffer.initial_hex)?;
        let mut backing = vec![guard; usize::try_from(buffer.allocation_size)?];
        let start = usize::try_from(buffer.offset)?;
        backing[start..start + initial.len()].copy_from_slice(&initial);
        allocations.push((buffer.allocation, backing));
        resources.insert_allocation(AllocationRecord {
            allocation_id: AllocationId::new(buffer.allocation),
            owner_epoch: provider.device_epoch(),
            size: buffer.allocation_size,
        })?;
        views.push(BufferView {
            view_id: ViewId::new(buffer.view),
            metal_binding: buffer.binding,
            allocation_id: AllocationId::new(buffer.allocation),
            offset: buffer.offset,
            length: buffer.length,
            access,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(initial),
        });
    }
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(operation),
        function: pipeline.function,
        pipeline_contract: pipeline.contract,
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![ComputePass {
            pipeline: pipeline.pipeline_id,
            buffers: views,
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: case.grid,
                threads_per_threadgroup: case.local,
            },
        }],
        completion_policy: CompletionPolicy::HostReadback,
    };
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .map_err(|error| format!("admit {}: {error:?}", case.id))?;
    let output = provider
        .submit(admitted)
        .map_err(|error| format!("submit {}: {error:?}", case.id))?;
    output.validate_for_trace(&trace)?;
    let CompletionDisposition::CompletedVisible { token } = output.completion else {
        return Err("provider capture requires completed visible results".into());
    };
    if provider
        .wait(token, Duration::ZERO)
        .map_err(|error| format!("wait: {error:?}"))?
        != output.completion
    {
        return Err("provider completion observation changed".into());
    }
    let mut writebacks = Vec::new();
    for write in output.writebacks {
        let (_, backing) = allocations
            .iter_mut()
            .find(|(id, _)| *id == write.allocation_id.get())
            .ok_or("unknown writeback allocation")?;
        let start = usize::try_from(write.offset)?;
        backing[start..start + write.bytes.len()].copy_from_slice(&write.bytes);
        writebacks.push(Writeback {
            allocation: write.allocation_id.get(),
            view: write.view_id.get(),
            offset: write.offset,
            bytes_hex: hex(&write.bytes),
        });
    }
    provider
        .release_completion(token)
        .map_err(|error| format!("release completion: {error:?}"))?;
    provider
        .release_pipeline(pipeline.pipeline_id)
        .map_err(|error| format!("release pipeline: {error:?}"))?;
    allocations.sort_by_key(|(id, _)| *id);
    Ok(CaseResult {
        id: case.id.clone(),
        completion: "CompletedVisible",
        writebacks,
        allocations: allocations
            .into_iter()
            .map(|(allocation, bytes)| Allocation {
                allocation,
                bytes_hex: hex(&bytes),
            })
            .collect(),
    })
}

fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(format!("input exceeds size limit: {}", path.display()).into());
    }
    Ok(bytes)
}

fn verified_source(directory: &Path, source: &Source) -> Result<Vec<u8>> {
    let path = directory.join(&source.path);
    let bytes = read_bounded(&path, MAX_BYTES)?;
    if hex(&Sha256::digest(&bytes)) != source.sha256 {
        return Err(format!("source digest mismatch: {}", path.display()).into());
    }
    Ok(bytes)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn unhex(value: &str) -> Result<Vec<u8>> {
    if !value.len().is_multiple_of(2)
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("hex data must be lowercase, with two digits per byte".into());
    }
    Ok(value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let digit = |b: u8| if b <= b'9' { b - b'0' } else { b - b'a' + 10 };
            digit(pair[0]) * 16 + digit(pair[1])
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn suite() -> Suite {
        serde_json::from_str(include_str!("../../../../conformance/suite.json")).unwrap()
    }
    #[test]
    fn suite_rejects_ranges_and_data_that_cannot_describe_the_shared_case() {
        let mut s = suite();
        validate_suite(&s).unwrap();
        s.cases[0].buffers[0].offset = u64::MAX;
        assert!(validate_suite(&s).is_err());
        let mut s = suite();
        s.cases[0].buffers[0].initial_hex = "00".into();
        assert!(validate_suite(&s).is_err());
        let mut s = suite();
        s.cases[0].buffers[0].allocation_size = u64::MAX;
        assert!(validate_suite(&s).is_err());
    }
    #[test]
    fn suite_cannot_silently_expand_the_qualified_shader_dispatch() {
        let mut s = suite();
        s.cases[1].grid = [11, 3, 1];
        assert!(validate_suite(&s).is_err());
        let mut s = suite();
        s.cases[1].id = s.cases[0].id.clone();
        assert!(validate_suite(&s).is_err());
        let mut s = suite();
        s.cases[0].air.sha256 = "0".repeat(64);
        assert!(validate_suite(&s).is_err());
        assert!(unhex("0aFF").is_err());
        assert!(unhex("0").is_err());
    }
}
