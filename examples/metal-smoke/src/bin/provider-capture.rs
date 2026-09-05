//! Capture a provider run of the shared, versioned native-oracle suite.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, BufferAccess, BufferSource, BufferView,
    CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass, ComputeTrace,
    Dispatch, DispatchKind, DispatchType, FootprintProof, OperationId, PipelineCompileRequest,
    PipelineProvider, ResourceTableSnapshot, SemanticDigest, ShaderSource, ViewId,
    PROVIDER_SCHEMA_VERSION,
};
#[cfg(target_os = "macos")]
use metal_api_native::NativeMetalProvider;
use metal_api_vulkan::VulkanComputeProvider;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const MAX_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    Vulkan,
    NativeMetalProvider,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::Vulkan => "vulkan",
            Self::NativeMetalProvider => "native-metal-provider",
        }
    }
}

fn create_provider(backend: Backend) -> Result<(Box<dyn PipelineProvider>, String)> {
    match backend {
        Backend::Vulkan => {
            let provider = VulkanComputeProvider::new()
                .map_err(|error| format!("create Vulkan provider: {error:?}"))?;
            let name = provider.device_name().to_owned();
            Ok((Box::new(provider), name))
        }
        Backend::NativeMetalProvider => {
            #[cfg(target_os = "macos")]
            {
                let provider = NativeMetalProvider::new()
                    .map_err(|error| format!("create native Metal provider: {error:?}"))?;
                let name = provider.device_name().to_owned();
                Ok((Box::new(provider), name))
            }
            #[cfg(not(target_os = "macos"))]
            Err("native-metal-provider requires macOS".into())
        }
    }
}

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
    dispatches: Option<Vec<CaseDispatch>>,
    programs: Option<Vec<CaseProgram>>,
}

#[derive(Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CaseDispatch {
    grid: [u64; 3],
    local: [u64; 3],
    bindings: Option<Vec<u64>>,
    program: Option<usize>,
}

#[derive(Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CaseProgram {
    entry: String,
    air: Source,
    metal: Source,
}

#[derive(Clone, Deserialize, PartialEq, Eq)]
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
    let mut backend = None;
    while let Some(flag) = args.next() {
        if flag == "--help" {
            println!(
                "usage: provider-capture --suite conformance/suite.json [--output capture.json] \
                 [--backend vulkan|native-metal-provider]"
            );
            return Ok(());
        }
        if flag == "--backend" && backend.is_none() {
            backend = Some(
                match args.next().as_deref().and_then(|value| value.to_str()) {
                    Some("vulkan") => Backend::Vulkan,
                    Some("native-metal-provider") => Backend::NativeMetalProvider,
                    _ => return Err("--backend requires vulkan or native-metal-provider".into()),
                },
            );
            continue;
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
    let backend = backend.unwrap_or(Backend::Vulkan);
    let suite_path = suite_path.ok_or("--suite is required")?;
    if output_path.as_ref().is_some_and(|path| path.exists()) {
        return Err("refusing to overwrite an existing capture".into());
    }
    // Validate every source and case before creating either provider device.
    let raw = read_bounded(&suite_path, 65536)?;
    let suite: Suite = serde_json::from_slice(&raw)?;
    validate_suite(&suite)?;
    let directory = suite_path.parent().unwrap_or(Path::new("."));
    let mut sources = BTreeMap::new();
    for case in &suite.cases {
        for program in case_programs(case) {
            let air = verified_source(directory, &program.air)?;
            let metal = verified_source(directory, &program.metal)?;
            sources.insert(
                program.entry,
                match backend {
                    Backend::Vulkan => ShaderSource::SanitizedLl(String::from_utf8(air)?),
                    Backend::NativeMetalProvider => {
                        ShaderSource::MetalSource(String::from_utf8(metal)?)
                    }
                },
            );
        }
    }
    let identity = hex(&Sha256::digest(&raw));
    let (provider, device_name) = create_provider(backend)?;
    let mut results = Vec::new();
    let mut pipelines = BTreeMap::new();
    for (entry, source) in sources {
        let pipeline = provider
            .compile(PipelineCompileRequest {
                entry_name: entry.clone(),
                logical_digest: SemanticDigest::new(
                    "suite-sha256-entry-v1",
                    format!("{identity}:{entry}").into_bytes(),
                )?,
                source,
            })
            .map_err(|error| format!("compile {entry}: {error:?}"))?;
        if backend == Backend::Vulkan && (entry == "transform_3d" || entry == "mix_3d") {
            verify_transform_contract(&pipeline)?;
        }
        eprintln!("{} artifact registered: entry={entry}", backend.name());
        pipelines.insert(entry, pipeline);
    }
    for (index, case) in suite.cases.iter().enumerate() {
        let programs = case_programs(case)
            .iter()
            .map(|program| pipelines[&program.entry].clone())
            .collect::<Vec<_>>();
        results.push(run_case(
            provider.as_ref(),
            &programs,
            case,
            index as u64 + 1,
            suite.guard_byte,
        )?);
    }
    for pipeline in pipelines.values() {
        provider
            .release_pipeline(pipeline)
            .map_err(|error| format!("release pipeline: {error:?}"))?;
    }
    let capture = Capture {
        schema_version: 1,
        suite: suite.suite,
        suite_sha256: identity,
        backend: backend.name(),
        allocation_observation: "host-writeback-landing",
        device: device_name,
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
    let case_ids: &[&str] = match (suite.schema_version, suite.suite.as_str()) {
        (1, "compute-buffer-v1") => &["copy_word", "indexed_boundary"],
        (1, "compute-buffer-v2") => &[
            "copy_seed_a",
            "copy_seed_b",
            "indexed_tail",
            "indexed_full",
            "indexed_small_grid",
            "indexed_unit",
            "transform_tail",
            "transform_small_grid",
        ],
        (1, "compute-buffer-v3") => &[
            "transform_twice",
            "transform_three_times",
            "transform_eight_times",
        ],
        (1, "compute-buffer-v4") => &[
            "transform_pingpong_two",
            "transform_pingpong_three",
            "transform_pingpong_eight",
            "copy_pingpong",
        ],
        (1, "compute-buffer-v5") => &[
            "pipeline_chain_two",
            "pipeline_chain_three",
            "pipeline_chain_eight",
        ],
        _ => return Err("unsupported suite identity/version".into()),
    };
    if suite.cases.len() != case_ids.len()
        || suite
            .cases
            .iter()
            .any(|case| !case_ids.contains(&case.id.as_str()))
    {
        return Err("incorrect case set for suite".into());
    }
    let mut ids = BTreeSet::new();
    for case in &suite.cases {
        validate_case_dispatches(case)?;
        validate_case_programs(case)?;
        if !ids.insert(&case.id) {
            return Err("duplicate case identity".into());
        }
        let (entry, grid, local, buffers) = case_shape(&case.id)?;
        if case.entry != entry
            || case.grid != grid
            || case.local != local
            || case.buffers.len() != buffers.len()
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
            if buffer.binding != buffers[index].0
                || buffer.access != buffers[index].1
                || buffer.length != buffers[index].2
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
        let mut writable: Vec<_> = case
            .buffers
            .iter()
            .filter(|b| ever_writable(case).contains(&b.view))
            .collect();
        writable.sort_by_key(|buffer| (buffer.allocation, buffer.view));
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

fn case_programs(case: &Case) -> Vec<CaseProgram> {
    case.programs.clone().unwrap_or_else(|| {
        vec![CaseProgram {
            entry: case.entry.clone(),
            air: case.air.clone(),
            metal: case.metal.clone(),
        }]
    })
}
fn validate_program(program: &CaseProgram) -> Result<()> {
    let (air_path, air_hash, metal_path, metal_hash) = match program.entry.as_str() {
        "copy_word" => (
            "../examples/metal-smoke/shaders/kernel_copy_word.ll",
            "292c3e1ff300fd08bf5e39aaa9abe352842eced807138f863e05056f39c56d99",
            "shaders/copy_word.metal",
            "7bfa419aef6eb0abcbec045c1bc15651b2d8f0a7591e07448edc6de6522141bc",
        ),
        "kernel_dispatch_threads_boundary_barrier" => (
            "../examples/metal-smoke/shaders/kernel_dispatch_threads_boundary_barrier.ll",
            "95076cf4199734f848fd6d761dce13addc7b55354b4d8ee2be16e59287ea5945",
            "shaders/indexed_boundary.metal",
            "7684e493a8704127e39dace5476a006fac564224909c667a57fb5ac9d8291b06",
        ),
        "transform_3d" => (
            "shaders/transform_3d.ll",
            "32bb9a29fef9825972b61cb982106b2bcb7c582413e50350eabc7834532b4df2",
            "shaders/transform_3d.metal",
            "5637cf50a3de44568ff7d3b09341e84111e2a9f6ff9b617181c6368efeacaf9b",
        ),
        "mix_3d" => (
            "shaders/mix_3d.ll",
            "cccc601c6f14d5c76808f927118d77cdcb9e4824591c0492faf735197afaf95f",
            "shaders/mix_3d.metal",
            "e3fa76b0027e6d20e4649fb6e7c07c0ca1618a9ae88fa13815337d2aa7c99bf5",
        ),
        _ => return Err("unknown shader entry".into()),
    };
    if program.air.path != air_path
        || program.air.sha256 != air_hash
        || program.metal.path != metal_path
        || program.metal.sha256 != metal_hash
    {
        return Err("unreviewed shader identity".into());
    }
    Ok(())
}
fn validate_case_programs(case: &Case) -> Result<()> {
    let programs = case_programs(case);
    if case.id.starts_with("pipeline_chain_") {
        if case.programs.is_none()
            || programs.len() != 2
            || programs[0].entry != "transform_3d"
            || programs[1].entry != "mix_3d"
            || programs[0].entry != case.entry
            || programs[0].air != case.air
            || programs[0].metal != case.metal
        {
            return Err("unreviewed program table".into());
        }
    } else if case.programs.is_some() {
        return Err("legacy fixture cannot carry program table".into());
    }
    for program in programs {
        validate_program(&program)?;
    }
    Ok(())
}

type CaseShape = (
    &'static str,
    [u64; 3],
    [u64; 3],
    &'static [(u32, &'static str, u64)],
);

fn case_shape(id: &str) -> Result<CaseShape> {
    let copy = (
        "copy_word",
        [1, 1, 1],
        [1, 1, 1],
        &[(0, "read", 4), (1, "write", 4)][..],
    );
    let indexed = |local| {
        (
            "kernel_dispatch_threads_boundary_barrier",
            [10, 3, 1],
            local,
            &[(0, "write", 120)][..],
        )
    };
    let transform = |local| {
        (
            "transform_3d",
            [5, 3, 2],
            local,
            &[(0, "read_write", 120), (2, "read", 4), (5, "write", 120)][..],
        )
    };
    Ok(match id {
        "copy_word" | "copy_seed_a" | "copy_seed_b" | "copy_pingpong" => copy,
        "indexed_boundary" | "indexed_tail" => indexed([8, 2, 1]),
        "indexed_full" => indexed([5, 3, 1]),
        "indexed_small_grid" => indexed([16, 4, 1]),
        "indexed_unit" => indexed([1, 1, 1]),
        "transform_tail"
        | "transform_twice"
        | "transform_three_times"
        | "transform_eight_times"
        | "transform_pingpong_two"
        | "transform_pingpong_three"
        | "transform_pingpong_eight"
        | "pipeline_chain_two"
        | "pipeline_chain_three"
        | "pipeline_chain_eight" => transform([4, 2, 2]),
        "transform_small_grid" => transform([8, 4, 4]),
        _ => return Err("unknown case identity".into()),
    })
}

fn validate_case_dispatches(case: &Case) -> Result<()> {
    let count = match case.id.as_str() {
        "transform_twice" | "transform_pingpong_two" | "copy_pingpong" | "pipeline_chain_two" => 2,
        "transform_three_times" | "transform_pingpong_three" | "pipeline_chain_three" => 3,
        "transform_eight_times" | "transform_pingpong_eight" | "pipeline_chain_eight" => 8,
        _ => {
            if case.dispatches.is_some() {
                return Err("single-pass fixture cannot carry a sequence".into());
            }
            return Ok(());
        }
    };
    let dispatches = case
        .dispatches
        .as_ref()
        .ok_or("sequence fixture requires dispatches")?;
    if dispatches.len() != count {
        return Err("wrong sequence dispatch count".into());
    }
    let locals = [[4, 2, 2], [8, 4, 4], [1, 1, 1]];
    let mixed = case.id.starts_with("pipeline_chain_");
    let pingpong = case.id.contains("pingpong") || mixed;
    for (i, dispatch) in dispatches.iter().enumerate() {
        if dispatch.program != mixed.then_some(i % 2) {
            return Err("unreviewed program selection".into());
        }
        let (grid, local) = if case.id == "copy_pingpong" {
            ([1, 1, 1], [1, 1, 1])
        } else {
            ([5, 3, 2], locals[i % locals.len()])
        };
        if dispatch.grid != grid || dispatch.local != local {
            return Err("unreviewed sequence dispatch shape".into());
        }
        if pingpong {
            let mut expected: Vec<_> = case.buffers.iter().map(|buffer| buffer.view).collect();
            let last = if case.id == "copy_pingpong" { 1 } else { 2 };
            if expected.len() <= last {
                return Err("missing pingpong resource".into());
            }
            if i % 2 == 1 {
                expected.swap(0, last);
            }
            if dispatch.bindings.as_ref() != Some(&expected) {
                return Err("unreviewed pingpong binding map".into());
            }
        } else if dispatch.bindings.is_some() {
            return Err("non-rebinding fixture cannot carry binding maps".into());
        }
    }
    if case.grid != dispatches[0].grid || case.local != dispatches[0].local {
        return Err("sequence first dispatch does not match case".into());
    }
    Ok(())
}

fn ever_writable(case: &Case) -> BTreeSet<u64> {
    if let Some(dispatches) = &case.dispatches {
        dispatches
            .iter()
            .flat_map(|dispatch| {
                case.buffers
                    .iter()
                    .enumerate()
                    .filter(|(_, slot)| slot.access != "read")
                    .map(|(index, slot)| {
                        dispatch
                            .bindings
                            .as_ref()
                            .map_or(slot.view, |map| map[index])
                    })
            })
            .collect()
    } else {
        case.buffers
            .iter()
            .filter(|b| b.access != "read")
            .map(|b| b.view)
            .collect()
    }
}

fn verify_transform_contract(pipeline: &CompiledComputePipeline) -> Result<()> {
    let bindings = &pipeline.contract.buffer_bindings;
    if bindings
        .iter()
        .map(|b| (b.metal_binding, b.access))
        .collect::<Vec<_>>()
        != [
            (0, BufferAccess::ReadWrite),
            (2, BufferAccess::Read),
            (5, BufferAccess::Write),
        ]
    {
        return Err("3D fixture sparse/access reflection mismatch".into());
    }
    for binding in [&bindings[0], &bindings[2]] {
        let FootprintProof::Affine { accesses } = &binding.footprint else {
            return Err("3D fixture must carry an affine footprint".into());
        };
        if accesses.is_empty() {
            return Err("3D fixture has no proven accesses".into());
        }
        for access in accesses {
            let mut strides = [0u64; 3];
            for term in &access.terms {
                let slot = strides
                    .get_mut(usize::from(term.axis))
                    .ok_or("3D fixture unknown axis")?;
                *slot = slot
                    .checked_add(term.stride)
                    .ok_or("3D fixture stride overflow")?;
            }
            if access.base_offset != 0 || access.access_size != 4 || strides != [4, 20, 60] {
                return Err("3D fixture footprint must prove 120-byte XYZ reach".into());
            }
        }
    }
    if bindings[1].footprint != (FootprintProof::Static { max_bytes: 4 }) {
        return Err("3D fixture scalar bias reach mismatch".into());
    }
    Ok(())
}

fn run_case(
    provider: &dyn PipelineProvider,
    programs: &[CompiledComputePipeline],
    case: &Case,
    operation: u64,
    guard: u8,
) -> Result<CaseResult> {
    let pipeline = &programs[0];
    let mut resources = ResourceTableSnapshot::new();
    let mut views = Vec::new();
    let mut allocations = Vec::new();
    for buffer in &case.buffers {
        let access = match buffer.access.as_str() {
            "read" => BufferAccess::Read,
            "write" => BufferAccess::Write,
            "read_write" => BufferAccess::ReadWrite,
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
        pipelines: programs.to_vec(),
        encoder_dispatch_type: DispatchType::Serial,
        passes: case
            .dispatches
            .clone()
            .unwrap_or_else(|| {
                vec![CaseDispatch {
                    grid: case.grid,
                    local: case.local,
                    bindings: None,
                    program: None,
                }]
            })
            .into_iter()
            .map(|dispatch| {
                let selected = &programs[dispatch.program.unwrap_or(0)];
                let buffers = views
                    .iter()
                    .enumerate()
                    .map(|(index, slot)| {
                        let view_id = dispatch
                            .bindings
                            .as_ref()
                            .map_or(slot.view_id.get(), |map| map[index]);
                        let mut resource = views
                            .iter()
                            .find(|view| view.view_id.get() == view_id)
                            .expect("validated binding map")
                            .clone();
                        resource.metal_binding = slot.metal_binding;
                        resource.access = slot.access;
                        resource
                    })
                    .collect();
                ComputePass {
                    pipeline: selected.pipeline_id,
                    buffers,
                    dispatch: Dispatch {
                        kind: DispatchKind::ThreadsExact,
                        grid: dispatch.grid,
                        threads_per_threadgroup: dispatch.local,
                    },
                }
            })
            .collect(),
        completion_policy: CompletionPolicy::HostReadback,
    };
    if case.entry == "transform_3d" {
        let mut short = trace.clone();
        let shortened = short.passes[0].buffers[0].view_id;
        for pass in &mut short.passes {
            let view = pass
                .buffers
                .iter_mut()
                .find(|view| view.view_id == shortened)
                .unwrap();
            view.length = 119;
            if let BufferSource::OwnedBytes(bytes) = &mut view.source {
                bytes.truncate(119);
            }
        }
        let rejected = provider.capabilities().admit(&short, &resources);
        if !matches!(rejected, Err(ref error) if error.slug == "buffer_footprint_exceeds_view") {
            return Err(format!("3D fixture must refuse 119-byte view: {rejected:?}").into());
        }
    }
    if programs.len() > 1 {
        let mut forged = trace.clone();
        forged.pipelines[1].contract.buffer_bindings[0].footprint =
            FootprintProof::Static { max_bytes: 1 };
        let input = provider
            .capabilities()
            .validate_trace(forged, resources.clone())
            .map_err(|error| format!("malformed late-pipeline refusal fixture: {error:?}"))?;
        if !matches!(provider.submit(input), Err(error) if error.slug == "pipeline_contract_mismatch"
            && error.completion == CompletionDisposition::NotSubmitted)
        {
            return Err(
                "provider failed to reject second-pipeline forged metadata before submission"
                    .into(),
            );
        }
        let mut unknown = trace.clone();
        let original_id = unknown.pipelines[1].pipeline_id;
        let missing = metal_api_core::provider::PipelineId::new(u64::MAX);
        unknown.pipelines[1].pipeline_id = missing;
        for pass in &mut unknown.passes {
            if pass.pipeline == original_id {
                pass.pipeline = missing;
            }
        }
        let input = provider
            .capabilities()
            .validate_trace(unknown, resources.clone())
            .map_err(|error| format!("malformed unknown-pipeline refusal fixture: {error:?}"))?;
        if !matches!(provider.submit(input), Err(error) if error.slug == "unknown_pipeline"
            && error.completion == CompletionDisposition::NotSubmitted)
        {
            return Err(
                "provider failed to reject unknown second pipeline before submission".into(),
            );
        }
        eprintln!("Checked second-pipeline refusal guards: {}", case.id);
    }
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

    #[test]
    fn v2_requires_sparse_bindings_read_write_access_and_ordered_outputs() {
        let load = || {
            serde_json::from_str::<Suite>(include_str!("../../../../conformance/suite-v2.json"))
                .unwrap()
        };
        let s = load();
        validate_suite(&s).unwrap();
        let mut s = load();
        s.cases[6].buffers[1].binding = 1;
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[6].buffers[0].access = "write".into();
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[6].expected_writebacks.reverse();
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[6].buffers[0].length = 119;
        assert!(validate_suite(&s).is_err());
    }

    #[test]
    fn v2_cases_cannot_be_mislabeled_as_v1_or_change_fixed_3d_shape() {
        let load = || {
            serde_json::from_str::<Suite>(include_str!("../../../../conformance/suite-v2.json"))
                .unwrap()
        };
        let mut s = load();
        s.suite = "compute-buffer-v1".into();
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[7].grid = [6, 3, 2];
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[0].id = "copy_word".into();
        assert!(validate_suite(&s).is_err());
    }

    #[test]
    fn serial_suite_admits_only_reviewed_dispatch_sequences() {
        let load = || {
            serde_json::from_str::<Suite>(include_str!("../../../../conformance/suite-v3.json"))
                .unwrap()
        };
        let s = load();
        validate_suite(&s).unwrap();
        let mut s = load();
        s.cases[0].dispatches = None;
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[0].dispatches.as_mut().unwrap().pop();
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[2].dispatches.as_mut().unwrap().push(CaseDispatch {
            grid: [5, 3, 2],
            local: [4, 2, 2],
            bindings: None,
            program: None,
        });
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[1].dispatches.as_mut().unwrap()[1].grid = [6, 3, 2];
        assert!(validate_suite(&s).is_err());
    }

    #[test]
    fn a_single_pass_case_cannot_silently_acquire_extra_gpu_work() {
        let mut s = suite();
        s.cases[0].dispatches = Some(vec![CaseDispatch {
            grid: [1, 1, 1],
            local: [1, 1, 1],
            bindings: None,
            program: None,
        }]);
        assert!(validate_suite(&s).is_err());
    }

    #[test]
    fn pingpong_case_requires_exact_view_permutations_and_final_writebacks() {
        let load = || {
            serde_json::from_str::<Suite>(include_str!("../../../../conformance/suite-v4.json"))
                .unwrap()
        };
        let s = load();
        validate_suite(&s).unwrap();
        assert_eq!(ever_writable(&s.cases[3]), BTreeSet::from([200, 201]));
        for map in [vec![410, 420, 410], vec![420, 410, 400], vec![410, 420]] {
            let mut s = load();
            s.cases[0].dispatches.as_mut().unwrap()[1].bindings = Some(map);
            assert!(validate_suite(&s).is_err());
        }
        let mut s = load();
        s.cases[3].expected_writebacks.pop();
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[0].dispatches.as_mut().unwrap()[1].bindings = None;
        assert!(validate_suite(&s).is_err());
    }

    #[test]
    fn mixed_suite_rejects_missing_unreviewed_or_unselected_programs() {
        let load = || {
            serde_json::from_str::<Suite>(include_str!("../../../../conformance/suite-v5.json"))
                .unwrap()
        };
        validate_suite(&load()).unwrap();
        let mut s = load();
        s.cases[0].programs = None;
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[0].programs.as_mut().unwrap()[1].metal.sha256 = "0".repeat(64);
        assert!(validate_suite(&s).is_err());
        for program in [None, Some(0), Some(2)] {
            let mut s = load();
            s.cases[0].dispatches.as_mut().unwrap()[1].program = program;
            assert!(validate_suite(&s).is_err());
        }
        let mut s = load();
        s.cases[0].programs.as_mut().unwrap().reverse();
        assert!(validate_suite(&s).is_err());
    }
}
