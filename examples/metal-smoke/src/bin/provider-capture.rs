//! Capture a provider run of the shared, versioned native-oracle suite.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, BufferAccess, BufferSource, BufferView,
    CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass, ComputeTrace,
    DeviceEpoch, Dispatch, DispatchKind, DispatchType, FootprintProof, OperationId,
    PipelineCompileRequest, PipelineProvider, ResourceTableSnapshot, SemanticDigest, ShaderSource,
    ViewId, PROVIDER_SCHEMA_VERSION,
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
    buffer_slots: Option<Vec<BufferSlot>>,
}

#[derive(Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct BufferSlot {
    binding: u32,
    access: String,
    length: u64,
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
        if backend == Backend::Vulkan && entry == "remap_3d" {
            let bindings = &pipeline.contract.buffer_bindings;
            if bindings
                .iter()
                .map(|b| (b.metal_binding, b.access))
                .collect::<Vec<_>>()
                != [
                    (1, BufferAccess::Read),
                    (3, BufferAccess::Read),
                    (7, BufferAccess::Write),
                ]
            {
                return Err("remap sparse layout/access reflection mismatch".into());
            }
            verify_xyz_access(&bindings[1].footprint)?;
            verify_xyz_access(&bindings[2].footprint)?;
            if bindings[0].footprint != (FootprintProof::Static { max_bytes: 4 }) {
                return Err("remap scalar bias reach mismatch".into());
            }
        }
        if backend == Backend::Vulkan && entry == "copy_3d" {
            verify_copy_contract(&pipeline)?;
        }
        eprintln!("{} artifact registered: entry={entry}", backend.name());
        pipelines.insert(entry, pipeline);
    }
    for (index, case) in suite.cases.iter().enumerate() {
        let programs = case_programs(case)
            .iter()
            .map(|program| pipelines[&program.entry].clone())
            .collect::<Vec<_>>();
        for (source, compiled) in case_programs(case).iter().zip(&programs) {
            if let Some(slots) = &source.buffer_slots {
                let expected = slots
                    .iter()
                    .map(|slot| {
                        (
                            slot.binding,
                            match slot.access.as_str() {
                                "read" => BufferAccess::Read,
                                "write" => BufferAccess::Write,
                                _ => BufferAccess::ReadWrite,
                            },
                        )
                    })
                    .collect::<Vec<_>>();
                if compiled
                    .contract
                    .buffer_bindings
                    .iter()
                    .map(|b| (b.metal_binding, b.access))
                    .collect::<Vec<_>>()
                    != expected
                {
                    return Err(
                        "compiled reflection differs from per-program fixture layout".into(),
                    );
                }
            }
        }
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
        (1, "compute-buffer-v6") => &[
            "layout_chain_two",
            "layout_chain_three",
            "layout_chain_eight",
        ],
        (1, "compute-buffer-v7") => &[
            "subset_chain_two",
            "subset_chain_four",
            "subset_chain_eight",
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
        validate_case_programs(case)?;
        validate_case_dispatches(case)?;
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
            buffer_slots: None,
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
        "remap_3d" => (
            "shaders/remap_3d.ll",
            "5388b13783b13a616a3b6952e0c939a120e5d1961e060dd15c11cb54083092ec",
            "shaders/remap_3d.metal",
            "0d715fe43e72fd96218f3fefc9a582c8634092fa10cc79a544869b5dee025a76",
        ),
        "copy_3d" => (
            "shaders/copy_3d.ll",
            "9f379575b8f9ed45e62df27c24761d0030e257f45c6241c649b5caae73cbe9cb",
            "shaders/copy_3d.metal",
            "3d8d71178abe03067508183a87f8c5c6843f1a3092e7f1cb52471ecaaaf0593f",
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
    let layout_change = case.id.starts_with("layout_chain_");
    let subsets = case.id.starts_with("subset_chain_");
    if subsets {
        let expected: &[&str] = if case.id == "subset_chain_two" {
            &["transform_3d", "copy_3d"]
        } else {
            &["transform_3d", "copy_3d", "remap_3d"]
        };
        if case.programs.is_none()
            || programs
                .iter()
                .map(|program| program.entry.as_str())
                .collect::<Vec<_>>()
                != expected
            || programs[0].entry != case.entry
            || programs[0].air != case.air
            || programs[0].metal != case.metal
        {
            return Err("unreviewed subset program table".into());
        }
    } else if case.id.starts_with("pipeline_chain_") || layout_change {
        if case.programs.is_none()
            || programs.len() != 2
            || programs[0].entry != "transform_3d"
            || programs[1].entry != if layout_change { "remap_3d" } else { "mix_3d" }
            || programs[0].entry != case.entry
            || programs[0].air != case.air
            || programs[0].metal != case.metal
        {
            return Err("unreviewed program table".into());
        }
    } else if case.programs.is_some() {
        return Err("legacy fixture cannot carry program table".into());
    }
    for (index, program) in programs.iter().enumerate() {
        validate_program(program)?;
        if layout_change || subsets {
            let expected = if index == 0 {
                vec![(0, "read_write", 120), (2, "read", 4), (5, "write", 120)]
            } else if subsets && index == 1 {
                vec![(4, "read", 120), (9, "write", 120)]
            } else {
                vec![(1, "read", 4), (3, "read", 120), (7, "write", 120)]
            };
            let expected = expected
                .into_iter()
                .map(|(binding, access, length)| BufferSlot {
                    binding,
                    access: access.into(),
                    length,
                })
                .collect::<Vec<_>>();
            if program.buffer_slots.as_ref() != Some(&expected) {
                return Err("unreviewed per-program layout".into());
            }
        } else if program.buffer_slots.is_some() {
            return Err("legacy program cannot declare a layout".into());
        }
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
        | "pipeline_chain_eight"
        | "layout_chain_two"
        | "layout_chain_three"
        | "layout_chain_eight" => transform([4, 2, 2]),
        "transform_small_grid" => transform([8, 4, 4]),
        "subset_chain_two" => (
            "transform_3d",
            [5, 3, 2],
            [4, 2, 2],
            &[
                (0, "read_write", 120),
                (2, "read", 4),
                (5, "write", 120),
                (8, "write", 120),
            ],
        ),
        "subset_chain_four" | "subset_chain_eight" => (
            "transform_3d",
            [5, 3, 2],
            [4, 2, 2],
            &[
                (0, "read_write", 120),
                (2, "read", 4),
                (5, "write", 120),
                (8, "write", 120),
                (9, "write", 120),
            ],
        ),
        _ => return Err("unknown case identity".into()),
    })
}

fn validate_case_dispatches(case: &Case) -> Result<()> {
    let count = match case.id.as_str() {
        "transform_twice"
        | "transform_pingpong_two"
        | "copy_pingpong"
        | "pipeline_chain_two"
        | "layout_chain_two"
        | "subset_chain_two" => 2,
        "subset_chain_four" => 4,
        "transform_three_times"
        | "transform_pingpong_three"
        | "pipeline_chain_three"
        | "layout_chain_three" => 3,
        "transform_eight_times"
        | "transform_pingpong_eight"
        | "pipeline_chain_eight"
        | "layout_chain_eight"
        | "subset_chain_eight" => 8,
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
    let layout_change = case.id.starts_with("layout_chain_");
    let subsets = case.id.starts_with("subset_chain_");
    let mixed = case.id.starts_with("pipeline_chain_") || layout_change;
    let pingpong = case.id.contains("pingpong") || mixed;
    for (i, dispatch) in dispatches.iter().enumerate() {
        let program = if subsets {
            Some([0, 1, 2, 1][i % 4])
        } else {
            mixed.then_some(i % 2)
        };
        if dispatch.program != program {
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
        if pingpong || subsets {
            let expected = if subsets {
                let indices: &[usize] = match i % 4 {
                    0 => &[0, 1, 2],
                    1 => &[2, 3],
                    2 => &[1, 3, 0],
                    _ => &[0, 4],
                };
                indices
                    .iter()
                    .map(|&index| {
                        case.buffers
                            .get(index)
                            .map(|b| b.view)
                            .ok_or("missing subset resource")
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?
            } else {
                let mut expected: Vec<_> = case.buffers.iter().map(|buffer| buffer.view).collect();
                let last = if case.id == "copy_pingpong" { 1 } else { 2 };
                if expected.len() <= last {
                    return Err("missing pingpong resource".into());
                }
                if i % 2 == 1 {
                    if layout_change {
                        expected.rotate_left(1);
                    } else {
                        expected.swap(0, last);
                    }
                }
                expected
            };
            if dispatch.bindings.as_ref() != Some(&expected) {
                return Err("unreviewed pingpong binding map".into());
            }
            for (slot, view_id) in selected_slots(case, dispatch).iter().zip(&expected) {
                let view = case
                    .buffers
                    .iter()
                    .find(|view| view.view == *view_id)
                    .ok_or("unknown mapped resource")?;
                if view.length != slot.length {
                    return Err("mapped resource extent differs from selected layout".into());
                }
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

fn selected_slots(case: &Case, dispatch: &CaseDispatch) -> Vec<BufferSlot> {
    if let Some(programs) = &case.programs {
        if let Some(slots) = &programs[dispatch.program.unwrap_or(0)].buffer_slots {
            return slots.clone();
        }
    }
    case.buffers
        .iter()
        .map(|b| BufferSlot {
            binding: b.binding,
            access: b.access.clone(),
            length: b.length,
        })
        .collect()
}

fn ever_writable(case: &Case) -> BTreeSet<u64> {
    if let Some(dispatches) = &case.dispatches {
        dispatches
            .iter()
            .flat_map(|dispatch| {
                selected_slots(case, dispatch)
                    .into_iter()
                    .enumerate()
                    .filter(|(_, slot)| slot.access != "read")
                    .map(|(i, _)| {
                        dispatch
                            .bindings
                            .as_ref()
                            .map_or(case.buffers[i].view, |map| map[i])
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
    verify_xyz_access(&bindings[0].footprint)?;
    verify_xyz_access(&bindings[2].footprint)?;
    if bindings[1].footprint != (FootprintProof::Static { max_bytes: 4 }) {
        return Err("3D fixture scalar bias reach mismatch".into());
    }
    Ok(())
}

fn verify_copy_contract(pipeline: &CompiledComputePipeline) -> Result<()> {
    let bindings = &pipeline.contract.buffer_bindings;
    if bindings
        .iter()
        .map(|binding| (binding.metal_binding, binding.access))
        .collect::<Vec<_>>()
        != [(4, BufferAccess::Read), (9, BufferAccess::Write)]
    {
        return Err("copy sparse layout/access reflection mismatch".into());
    }
    for binding in bindings {
        verify_xyz_access(&binding.footprint)?;
    }
    Ok(())
}

fn verify_xyz_access(footprint: &FootprintProof) -> Result<()> {
    let FootprintProof::Affine { accesses } = footprint else {
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
    Ok(())
}

fn case_trace(
    device_epoch: DeviceEpoch,
    programs: &[CompiledComputePipeline],
    case: &Case,
    operation: u64,
    views: &[BufferView],
) -> Result<ComputeTrace> {
    Ok(ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch,
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
            .map(|dispatch| -> Result<ComputePass> {
                let selected = &programs[dispatch.program.unwrap_or(0)];
                let expected = selected_slots(case, &dispatch);
                if selected.contract.buffer_bindings.len() != expected.len()
                    || selected.contract.buffer_bindings.iter().zip(&expected).any(
                        |(actual, expected)| {
                            actual.metal_binding != expected.binding
                                || expected.access
                                    != match actual.access {
                                        BufferAccess::Read => "read",
                                        BufferAccess::Write => "write",
                                        BufferAccess::ReadWrite => "read_write",
                                        BufferAccess::Unused => "unused",
                                    }
                        },
                    )
                {
                    return Err("source/fixture selected layout mismatch".into());
                }
                let buffers = selected
                    .contract
                    .buffer_bindings
                    .iter()
                    .enumerate()
                    .map(|(index, slot)| {
                        let view_id = dispatch
                            .bindings
                            .as_ref()
                            .map_or(views[index].view_id.get(), |map| map[index]);
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
                Ok(ComputePass {
                    pipeline: selected.pipeline_id,
                    buffers,
                    dispatch: Dispatch {
                        kind: DispatchKind::ThreadsExact,
                        grid: dispatch.grid,
                        threads_per_threadgroup: dispatch.local,
                    },
                })
            })
            .collect::<Result<_>>()?,
        completion_policy: CompletionPolicy::HostReadback,
    })
}

fn run_case(
    provider: &dyn PipelineProvider,
    programs: &[CompiledComputePipeline],
    case: &Case,
    operation: u64,
    guard: u8,
) -> Result<CaseResult> {
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
    let trace = case_trace(provider.device_epoch(), programs, case, operation, &views)?;
    if case.entry == "transform_3d" {
        let mut short = trace.clone();
        let shortened = short.passes[0].buffers[0].view_id;
        for pass in &mut short.passes {
            if let Some(view) = pass
                .buffers
                .iter_mut()
                .find(|view| view.view_id == shortened)
            {
                view.length = 119;
                if let BufferSource::OwnedBytes(bytes) = &mut view.source {
                    bytes.truncate(119);
                }
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

    #[test]
    fn differing_layout_requires_selected_slot_numbers_access_and_lengths() {
        let load = || {
            serde_json::from_str::<Suite>(include_str!("../../../../conformance/suite-v6.json"))
                .unwrap()
        };
        let s = load();
        validate_suite(&s).unwrap();
        let dispatch = &s.cases[0].dispatches.as_ref().unwrap()[1];
        let slots = selected_slots(&s.cases[0], dispatch);
        assert_eq!(
            slots
                .iter()
                .map(|slot| (slot.binding, slot.access.as_str(), slot.length))
                .collect::<Vec<_>>(),
            [(1, "read", 4), (3, "read", 120), (7, "write", 120)]
        );
        let mut s = load();
        s.cases[0].programs.as_mut().unwrap()[1].buffer_slots = None;
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[0].programs.as_mut().unwrap()[1]
            .buffer_slots
            .as_mut()
            .unwrap()[0]
            .binding = 0;
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[0].programs.as_mut().unwrap()[1]
            .buffer_slots
            .as_mut()
            .unwrap()[1]
            .access = "read_write".into();
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[0].programs.as_mut().unwrap()[1]
            .buffer_slots
            .as_mut()
            .unwrap()[0]
            .length = 120;
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[0].dispatches.as_mut().unwrap()[1].bindings = Some(vec![400, 420, 410]);
        assert!(validate_suite(&s).is_err());
    }

    fn subset_suite() -> Suite {
        serde_json::from_str(include_str!("../../../../conformance/suite-v7.json")).unwrap()
    }

    #[test]
    fn subset_suite_requires_reviewed_pool_sizes_and_dispatch_sequences() {
        let suite = subset_suite();
        validate_suite(&suite).unwrap();
        assert_eq!(
            suite
                .cases
                .iter()
                .map(|case| case.buffers.len())
                .collect::<Vec<_>>(),
            [4, 5, 5]
        );
        for (case_index, dispatch_index, mappings) in [
            (0, 1, vec![400, 410]),
            (1, 3, vec![410, 430]),
            (2, 4, vec![400, 420, 410]),
            (0, 1, vec![400, 430, 410]),
        ] {
            let mut invalid = subset_suite();
            invalid.cases[case_index].dispatches.as_mut().unwrap()[dispatch_index].bindings =
                Some(mappings);
            assert!(validate_suite(&invalid).is_err());
        }
        for program in [None, Some(0), Some(2), Some(usize::MAX)] {
            let mut invalid = subset_suite();
            invalid.cases[0].dispatches.as_mut().unwrap()[1].program = program;
            assert!(validate_suite(&invalid).is_err());
        }
        let mut invalid = subset_suite();
        invalid.cases[0].buffers.pop();
        assert!(validate_suite(&invalid).is_err());
        let mut invalid = subset_suite();
        invalid.cases[1].dispatches.as_mut().unwrap().pop();
        assert!(validate_suite(&invalid).is_err());
        let mut invalid = subset_suite();
        invalid.cases[2].dispatches.as_mut().unwrap()[7].local = [1, 1, 1];
        assert!(validate_suite(&invalid).is_err());
    }

    #[test]
    fn subset_programs_require_exact_sources_and_selected_slot_layouts() {
        let suite = subset_suite();
        let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance");
        for case in &suite.cases {
            for program in case_programs(case) {
                verified_source(&directory, &program.air).unwrap();
                verified_source(&directory, &program.metal).unwrap();
            }
        }
        let mut invalid = subset_suite();
        invalid.cases[0].programs.as_mut().unwrap()[1].metal.sha256 = "0".repeat(64);
        assert!(validate_suite(&invalid).is_err());
        let mut invalid = subset_suite();
        invalid.cases[0].programs.as_mut().unwrap()[1]
            .buffer_slots
            .as_mut()
            .unwrap()[0]
            .binding = 8;
        assert!(validate_suite(&invalid).is_err());
        let mut invalid = subset_suite();
        invalid.cases[0].programs.as_mut().unwrap()[1]
            .buffer_slots
            .as_mut()
            .unwrap()[0]
            .access = "read_write".into();
        assert!(validate_suite(&invalid).is_err());
        let mut invalid = subset_suite();
        invalid.cases[1].programs.as_mut().unwrap().pop();
        assert!(validate_suite(&invalid).is_err());
        let mut invalid = subset_suite();
        invalid.cases[0].buffers[3].length = 4;
        assert!(validate_suite(&invalid).is_err());
    }

    #[test]
    fn subset_expected_results_cover_late_and_temporarily_unbound_writes() {
        let suite = subset_suite();
        assert_eq!(
            ever_writable(&suite.cases[0]),
            BTreeSet::from([400, 410, 430])
        );
        assert_eq!(
            ever_writable(&suite.cases[1]),
            BTreeSet::from([400, 410, 430, 440])
        );
        for view in [400, 410, 430, 440] {
            let mut invalid = subset_suite();
            invalid.cases[1]
                .expected_writebacks
                .retain(|write| write.view != view);
            assert!(validate_suite(&invalid).is_err());
        }
    }

    fn fixture_pipelines(case: &Case) -> Vec<CompiledComputePipeline> {
        use metal_api_core::provider::{
            BufferBindingContract, FunctionIdentity, FunctionSource, PipelineContract, PipelineId,
        };
        case_programs(case)
            .into_iter()
            .enumerate()
            .map(|(index, program)| CompiledComputePipeline {
                device_epoch: DeviceEpoch::new(1),
                pipeline_id: PipelineId::new(index as u64 + 1),
                function: FunctionIdentity {
                    logical_digest: SemanticDigest::new("test", vec![1]).unwrap(),
                    entry_name: program.entry,
                    source: FunctionSource::MetalSource,
                },
                contract: PipelineContract {
                    dispatch_kind: DispatchKind::ThreadsExact,
                    required_local_size: None,
                    fixed_grid: Some(case.grid),
                    push_constant_offset: 0,
                    push_constant_bytes: 0,
                    buffer_bindings: program
                        .buffer_slots
                        .unwrap()
                        .into_iter()
                        .map(|slot| BufferBindingContract {
                            metal_binding: slot.binding,
                            access: match slot.access.as_str() {
                                "read" => BufferAccess::Read,
                                "write" => BufferAccess::Write,
                                "read_write" => BufferAccess::ReadWrite,
                                _ => unreachable!(),
                            },
                            footprint: FootprintProof::Static {
                                max_bytes: slot.length,
                            },
                        })
                        .collect(),
                    shader_capabilities: Vec::new(),
                    translator_revision: None,
                },
            })
            .collect()
    }

    #[test]
    fn subset_trace_retains_all_initial_resources_and_binds_only_selected_views() {
        let suite = subset_suite();
        validate_suite(&suite).unwrap();
        for case in &suite.cases {
            let mut programs = fixture_pipelines(case);
            let views = case
                .buffers
                .iter()
                .map(|buffer| BufferView {
                    view_id: ViewId::new(buffer.view),
                    allocation_id: AllocationId::new(buffer.allocation),
                    metal_binding: buffer.binding,
                    offset: buffer.offset,
                    length: buffer.length,
                    access: BufferAccess::Unused,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(unhex(&buffer.initial_hex).unwrap()),
                })
                .collect::<Vec<_>>();
            let trace = case_trace(DeviceEpoch::new(1), &programs, case, 1, &views).unwrap();
            let resources = trace.serial_resources().unwrap();
            assert_eq!(resources.len(), case.buffers.len());
            assert_eq!(trace.passes[0].buffers.len(), 3);
            assert_eq!(trace.passes[1].buffers.len(), 2);
            assert_eq!(
                trace.passes[1]
                    .buffers
                    .iter()
                    .map(|view| (view.metal_binding, view.view_id.get()))
                    .collect::<Vec<_>>(),
                [(4, 400), (9, 430)]
            );
            assert!(!trace.passes[1]
                .buffers
                .iter()
                .any(|view| view.view_id.get() == 410));
            for (initial, collected) in views.iter().zip(&resources) {
                assert_eq!(initial.view_id, collected.view_id);
                assert_eq!(initial.source, collected.source);
                assert_eq!(initial.offset, collected.offset);
            }
            assert_eq!(
                resources
                    .iter()
                    .filter(|view| view.access.is_writable())
                    .map(|view| view.view_id.get())
                    .collect::<BTreeSet<_>>(),
                ever_writable(case)
            );
            programs[1].contract.buffer_bindings[0].access = BufferAccess::ReadWrite;
            assert!(case_trace(DeviceEpoch::new(1), &programs, case, 1, &views).is_err());
        }
    }

    #[test]
    fn copy_contract_checks_both_sparse_accesses_and_xyz_reach() {
        use metal_api_core::provider::{AffineAccess, AffineTerm};
        let mut pipeline = fixture_pipelines(&subset_suite().cases[0]).remove(1);
        for binding in &mut pipeline.contract.buffer_bindings {
            binding.footprint = FootprintProof::Affine {
                accesses: vec![AffineAccess {
                    base_offset: 0,
                    access_size: 4,
                    terms: vec![
                        AffineTerm { axis: 0, stride: 4 },
                        AffineTerm {
                            axis: 1,
                            stride: 20,
                        },
                        AffineTerm {
                            axis: 2,
                            stride: 60,
                        },
                    ],
                }],
            };
        }
        verify_copy_contract(&pipeline).unwrap();
        for index in 0..2 {
            let mut invalid = pipeline.clone();
            invalid.contract.buffer_bindings[index].footprint =
                FootprintProof::Static { max_bytes: 120 };
            assert!(verify_copy_contract(&invalid).is_err());
        }
        pipeline.contract.buffer_bindings[0].metal_binding = 0;
        assert!(verify_copy_contract(&pipeline).is_err());
    }
}
