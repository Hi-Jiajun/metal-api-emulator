//! Capture a provider run of the shared, versioned native-oracle suite.

use metal_api_core::provider::queue_priorities_for_device;
#[cfg(unix)]
use metal_api_core::provider::ComputeProvider;
use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeTrace, DeviceEpoch, Dispatch, DispatchKind, DispatchType, FootprintProof, LoadOp,
    OperationId, PipelineCompileRequest, PipelineProvider, QueuePriority, QueueSchedulingPolicy,
    RenderAttachment, RenderPassDescriptor, RenderPipelineContract, ResourceTableSnapshot,
    SemanticDigest, ShaderSource, StoreOp, TextureAccess, TextureFormat, TextureSource,
    TextureType, TextureView, TracePass, VertexLayout, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{provider_api as objects, Size};
#[cfg(unix)]
use metal_api_ipc::command::{serve_provider_unix, unix as command_unix, RemoteProvider};
#[cfg(target_os = "macos")]
use metal_api_native::NativeMetalProvider;
use metal_api_vulkan::{RenderPipelineRequest, VulkanComputeProvider, VulkanExecutor};
use metal_smoke::{assemble_owned_air, wrap_air_bitcode};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, OpenOptions};
#[cfg(unix)]
use std::io::{BufRead, BufReader};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const MAX_BYTES: usize = 1024 * 1024;

/// The Vulkan rail's half of the reviewed render fixture
/// (`research/docs/23` §1.2). A render case pins the MSL module the canonical
/// rails compile, because that module is the review surface both the native
/// provider and the Swift oracle execute. This rail executes the same two
/// stages as SPIR-V, so their identity is pinned here in code, exactly as
/// `crates/metal-api-vulkan/src/render.rs` pins it in its own tests: a
/// re-hashed fixture must not be enough to admit different modules.
///
/// The entry names differ from the MSL ones (`render_fullscreen_triangle` /
/// `render_solid_rgba8`): the reviewed SPIR-V sources declare
/// `vertex_main` / `fragment_main`, and core refuses a render contract whose
/// two entries share a name.
const RENDER_VERTEX_SPV: &[u8] = include_bytes!(
    "../../../../crates/metal-api-vulkan/src/render_spv/fullscreen_triangle.vert.spv"
);
const RENDER_FRAGMENT_SPV: &[u8] =
    include_bytes!("../../../../crates/metal-api-vulkan/src/render_spv/solid_rgba8.frag.spv");
const RENDER_VERTEX_ENTRY: &str = "vertex_main";
const RENDER_FRAGMENT_ENTRY: &str = "fragment_main";

/// The capture backends a suite may declare a render case executable on. The
/// vocabulary is `conformance/compare.py`'s `ALLOCATION_OBSERVATIONS`, i.e. the
/// backends a capture reports: a marker cannot name a rail that no capture can
/// produce.
const RENDER_RAILS: &[&str] = &[
    "native-metal",
    "vulkan",
    "native-metal-provider",
    "vulkan-objects",
    "native-metal-provider-objects",
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    Vulkan,
    NativeMetalProvider,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EntryApi {
    Trace,
    Objects,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::Vulkan => "vulkan",
            Self::NativeMetalProvider => "native-metal-provider",
        }
    }

    fn report_name(self, api: EntryApi) -> &'static str {
        match (self, api) {
            (Self::Vulkan, EntryApi::Objects) => "vulkan-objects",
            (Self::NativeMetalProvider, EntryApi::Objects) => "native-metal-provider-objects",
            (_, EntryApi::Trace) => self.name(),
        }
    }
}

/// Where a capture reads its device-buffer copy counters. The rails take a
/// `dyn PipelineProvider`, so the concrete handle has to be kept here to read
/// the counters around each case (`research/docs/15` §5).
enum CopyCounters {
    Vulkan(Arc<VulkanExecutor>),
    #[cfg(target_os = "macos")]
    Native(Arc<NativeMetalProvider>),
}

/// What one capture run keeps hold of: the trait object every rail shares, the
/// device name, the copy counters, and the concrete Vulkan context the render
/// rail's `register_render_pipeline` entry point lives on (it is not part of
/// `PipelineProvider`).
type ProviderHandles = (
    Arc<dyn PipelineProvider>,
    String,
    CopyCounters,
    Option<Arc<VulkanComputeProvider>>,
);

impl CopyCounters {
    /// Cumulative (copy-in, copy-out) device-buffer operations.
    fn read(&self) -> (usize, usize) {
        match self {
            Self::Vulkan(executor) => executor.buffer_copy_counts(),
            #[cfg(target_os = "macos")]
            Self::Native(provider) => provider.buffer_copy_counts(),
        }
    }
}

/// The `research/docs/21` §6 queue marking: one high queue, one default queue
/// and six low queues.
fn default_queue_priorities() -> Vec<QueuePriority> {
    let mut tiers = vec![QueuePriority::Low; 8];
    tiers[0] = QueuePriority::High;
    tiers[1] = QueuePriority::Default;
    tiers
}

fn parse_queue_priorities(value: &str) -> Result<Vec<QueuePriority>> {
    value
        .split(',')
        .map(|entry| match entry.trim() {
            "low" => Ok(QueuePriority::Low),
            "default" => Ok(QueuePriority::Default),
            "high" => Ok(QueuePriority::High),
            other => Err(format!("unknown queue tier {other:?}; use low, default or high").into()),
        })
        .collect()
}

fn format_queue_tiers(tiers: &[QueuePriority]) -> String {
    tiers
        .iter()
        .map(|tier| match tier {
            QueuePriority::Low => "low",
            QueuePriority::Default => "default",
            QueuePriority::High => "high",
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// Install one scheduling tier per device queue, truncating or padding the
/// request so the table always describes the device exactly (Lavapipe exposes a
/// single queue, so the §6 marking degenerates to one high queue there). The
/// expansion is the core `queue_priorities_for_device`, i.e. the same rule a
/// provider applies to a marking that arrives over the command channel.
fn install_queue_priorities(executor: &VulkanExecutor, requested: &[QueuePriority]) -> Result<()> {
    let queues = executor.queue_count();
    let tiers = queue_priorities_for_device(queues, requested);
    executor
        .set_queue_priorities(&tiers)
        .map_err(|error| format!("install queue priorities: {error:?}"))?;
    println!(
        "queue_priority_probe device={} queues={} requested={} installed={} truncated={} padded={}",
        executor.device_name(),
        queues,
        format_queue_tiers(requested),
        format_queue_tiers(&tiers),
        requested.len().saturating_sub(queues),
        queues.saturating_sub(requested.len()),
    );
    Ok(())
}

/// `research/docs/21` §6 on a real device: mark the queues, submit `submissions`
/// mutually independent command buffers through the queue-selecting submit path
/// and report what both observation surfaces see.
///
/// Every command buffer is committed and retired before the next one is
/// recorded, so the device queues are idle at each selection and the policy
/// window — not the in-flight load — decides the tier. Lavapipe exposes one
/// queue: the probe still runs (the single-queue degenerate path) and skips the
/// window contract, which a one-tier device cannot show.
///
/// The probe submits through the deferred object path, which reports one probe
/// call per commit and therefore carries the whole selection sequence. The
/// synchronous paths select through the same policy, but they report one queue
/// per submit rather than a sequence, so the window contract is asserted here.
fn run_queue_priority_probe(executor: &Arc<VulkanExecutor>, submissions: usize) -> Result<()> {
    let installed = executor.queue_priorities();
    if installed.len() != executor.queue_count() {
        return Err("the installed queue priority table does not describe the device".into());
    }
    let observed = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&observed);
    executor.set_enqueue_probe_for_test(Arc::new(move |queue| {
        if let Ok(mut sequence) = sink.lock() {
            sequence.push(queue);
        }
    }));
    let provider: Arc<dyn PipelineProvider> = Arc::new(
        VulkanComputeProvider::with_executor(Arc::clone(executor))
            .map_err(|error| format!("create Vulkan provider: {error:?}"))?
            .with_async_execution(true),
    );
    drive_priority_probe(provider, submissions)?;
    executor.clear_enqueue_probe_for_test();
    let sequence = observed
        .lock()
        .map_err(|_| "the enqueue probe sequence is poisoned")?
        .clone();
    report_queue_priority_probe(executor, &installed, &sequence, submissions)
}

/// Submit `submissions` mutually independent command buffers through the
/// queue-selecting object path and assert the landing bytes.
///
/// The provider is a parameter so the same scenario drives a provider in this
/// process and a provider in another one (`--command-socket`); only the local
/// run can read the selection sequence back, because the enqueue probe is
/// host state inside the provider process.
fn drive_priority_probe(provider: Arc<dyn PipelineProvider>, submissions: usize) -> Result<()> {
    let device = objects::Device::new(provider);
    let pipeline = device.compile_pipeline(PipelineCompileRequest {
        entry_name: "read_texture_2d".to_owned(),
        logical_digest: SemanticDigest::new(
            "metal-smoke-fixture-v1",
            b"queue-priority-probe".to_vec(),
        )?,
        source: ShaderSource::SanitizedLl(
            include_str!("../../shaders/kernel_read_texture_2d.ll").to_owned(),
        ),
    })?;
    let mut texels = Vec::with_capacity(64);
    for value in 0..16_u32 {
        texels.extend_from_slice(&value.to_le_bytes());
    }
    let texture = device.new_texture_with_bytes(TextureFormat::R32Uint, 4, 4, texels)?;
    let output = device.new_buffer_with_bytes(vec![0_u8; 64])?;
    let queue = device.new_command_queue();
    for _ in 0..submissions {
        let command = queue.command_buffer();
        {
            let mut encoder = command.compute_command_encoder()?;
            encoder.set_compute_pipeline_state(&pipeline)?;
            encoder.set_texture(0, &texture)?;
            encoder.set_buffer(0, &output.view(0, 64)?)?;
            encoder.dispatch_threads(Size::new(1, 1, 1)?, Size::new(1, 1, 1)?)?;
            encoder.end_encoding()?;
        }
        command.commit()?;
        if command.status()? != metal_api_core::CommandBufferStatus::Committed {
            return Err("async commit did not leave the probe command pending".into());
        }
        command.wait_until_completed()?;
    }
    if output.read()?[..4] != 0_u32.to_le_bytes() {
        return Err("the queue priority probe landed unexpected bytes".into());
    }
    Ok(())
}

/// Cross-check both observation surfaces and the §6 assertions, printing one
/// machine-greppable PASS or SKIP line per contract.
fn report_queue_priority_probe(
    executor: &VulkanExecutor,
    installed: &[QueuePriority],
    sequence: &[usize],
    submissions: usize,
) -> Result<()> {
    let queues = installed.len();
    if sequence.len() != submissions {
        return Err(format!(
            "the enqueue probe observed {} of {submissions} submissions",
            sequence.len()
        )
        .into());
    }
    if let Some(index) = sequence.iter().find(|index| **index >= queues) {
        return Err(format!("the enqueue probe observed queue {index} outside the device").into());
    }
    let counts = executor.queue_submission_counts();
    if counts.len() != queues || counts.iter().sum::<usize>() != submissions {
        return Err(format!(
            "queue_submission_counts() reports {counts:?} for {submissions} submissions"
        )
        .into());
    }
    for (index, count) in counts.iter().enumerate() {
        let observed = sequence.iter().filter(|picked| **picked == index).count();
        if *count != observed {
            return Err(format!(
                "queue {index}: queue_submission_counts={count} enqueue_probe={observed}"
            )
            .into());
        }
    }

    let policy = QueueSchedulingPolicy::default();
    let tier_of = |index: usize| installed[index];
    let share = |tier: QueuePriority| {
        sequence
            .iter()
            .filter(|index| tier_of(**index) == tier)
            .count()
    };
    println!(
        "queue_priority_probe sequence={}",
        sequence
            .iter()
            .map(|index| match tier_of(*index) {
                QueuePriority::Low => 'L',
                QueuePriority::Default => 'D',
                QueuePriority::High => 'H',
            })
            .collect::<String>()
    );
    println!(
        "queue_priority_probe index_counts={}",
        counts
            .iter()
            .map(|count| count.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "queue_priority_probe tier_counts=high={} default={} low={}",
        share(QueuePriority::High),
        share(QueuePriority::Default),
        share(QueuePriority::Low)
    );
    println!(
        "PASS queue_priority_probe submissions={submissions} queues={queues} \
         probe_matches_counts=exact writeback=exact"
    );

    let window = usize::try_from(policy.window())?;
    let present = installed.iter().collect::<BTreeSet<_>>().len();
    if queues < window || present < 3 {
        println!(
            "SKIP queue_priority_probe_window reason=insufficient_queues queues={queues} \
             tiers_present={present} window={window}"
        );
        return Ok(());
    }
    let windows = submissions / window;
    if windows == 0 {
        return Err(
            format!("{submissions} submissions do not fill one {window}-slot window").into(),
        );
    }
    let tiers: Vec<QueuePriority> = sequence.iter().map(|index| tier_of(*index)).collect();
    let mut streak = 0_usize;
    let mut longest = 0_usize;
    for tier in &tiers {
        streak = if *tier == QueuePriority::High {
            streak + 1
        } else {
            0
        };
        longest = longest.max(streak);
    }
    let limit = policy.high_priority_streak_limit() as usize;
    if longest > limit {
        return Err(format!("the high tier ran {longest} times in a row, limit {limit}").into());
    }
    let mut low_per_window_min = usize::MAX;
    for window_tiers in tiers.chunks(window).take(windows) {
        let lows = window_tiers
            .iter()
            .filter(|tier| **tier == QueuePriority::Low)
            .count();
        low_per_window_min = low_per_window_min.min(lows);
    }
    if low_per_window_min == 0 {
        return Err("a window starved the low tier".into());
    }
    let expected_high = policy.high_weight() as usize * windows;
    let expected_default = policy.medium_weight() as usize * windows;
    let high = share(QueuePriority::High);
    let default = share(QueuePriority::Default);
    if high != expected_high || default != expected_default {
        return Err(format!(
            "window shares high={high} default={default} low={} do not match {expected_high}:{expected_default}",
            share(QueuePriority::Low)
        )
        .into());
    }
    println!(
        "PASS queue_priority_probe_window windows={windows} high={high} default={default} \
         low={} max_high_streak={longest} limit={limit} low_per_window_min={low_per_window_min}",
        share(QueuePriority::Low)
    );
    Ok(())
}

/// Log one selection sequence as the tier letter of each selected queue.
#[cfg(unix)]
fn format_queue_sequence(installed: &[QueuePriority], sequence: &[usize]) -> String {
    sequence
        .iter()
        .map(|index| match installed[*index] {
            QueuePriority::Low => 'L',
            QueuePriority::Default => 'D',
            QueuePriority::High => 'H',
        })
        .collect()
}

/// `--queue-priority-probe --command-socket <path>`: the probe scenario with
/// the provider in another process.
///
/// The owner sets the marking, drives the same submissions the in-process probe
/// drives, and compares the provider's own report of the tiers it read against
/// the table its marking installed. The enqueue probe is host state inside the
/// provider process, so the provider prints its selection sequence itself; the
/// owner asserts the two sides agree before this returns.
#[cfg(unix)]
fn run_remote_queue_priority_probe(
    command_path: &Path,
    requested: &[QueuePriority],
    submissions: usize,
) -> Result<()> {
    use metal_api_ipc::command::unix::UnixListenerCommandTransport;

    let listener = UnixListenerCommandTransport::bind(command_path)?;
    let mut child = Command::new(std::env::current_exe()?)
        .arg("--queue-priority-child")
        .arg(command_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or("the queue priority child stdout was not piped")?;
    let mut lines = BufReader::new(stdout).lines();

    let transport = match listener.accept() {
        Ok(transport) => transport,
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("accept the queue priority child: {error}").into());
        }
    };
    let remote = RemoteProvider::connect(transport)?;
    // The response is the provider's own expansion of the marking, so this is
    // the table the scheduler in the child process reads.
    let installed = remote
        .set_queue_priorities(requested)
        .map_err(|error| format!("install remote queue priorities: {error:?}"))?;
    println!(
        "queue_priority_probe owner=remote child=command-socket sent={} installed={} queues={}",
        format_queue_tiers(requested),
        format_queue_tiers(&installed),
        installed.len()
    );
    // Dropping the device closes the command channel, which is how the child
    // learns that the session is over.
    drive_priority_probe(Arc::new(remote) as Arc<dyn PipelineProvider>, submissions)?;

    let mut child_lines = Vec::new();
    for line in &mut lines {
        child_lines.push(line?);
    }
    let status = child.wait()?;
    let _ = std::fs::remove_file(command_path);
    for line in &child_lines {
        println!("child: {line}");
    }
    if !status.success() {
        return Err(format!("the queue priority child exited with {status}").into());
    }

    // Owner-side assertion of the provider-side read: the child reports the
    // table it installed, and it has to be the table this marking installed.
    let report = child_lines
        .iter()
        .find(|line| line.starts_with("queue_priority_child read="))
        .ok_or("the queue priority child did not report the tiers it read")?;
    let field = |name: &str| {
        report
            .split_whitespace()
            .find_map(|part| part.strip_prefix(name))
    };
    let read = field("read=").ok_or("the child report has no read= field")?;
    let queues = field("queues=").ok_or("the child report has no queues= field")?;
    if read != format_queue_tiers(&installed) || queues != installed.len().to_string() {
        return Err(format!(
            "the provider read {read} on {queues} queues, the owner installed {} on {}",
            format_queue_tiers(&installed),
            installed.len()
        )
        .into());
    }
    if !child_lines
        .iter()
        .any(|line| line.starts_with("PASS queue_priority_child"))
    {
        return Err("the queue priority child did not report a passing observation".into());
    }
    println!(
        "PASS queue_priority_probe_remote sent={} read={} queues={} submissions={submissions} \
         probe_matches_counts=exact writeback=exact",
        format_queue_tiers(requested),
        read,
        installed.len()
    );
    Ok(())
}

/// `--queue-priority-probe --command-socket <path>` on a platform without Unix
/// domain sockets.
#[cfg(not(unix))]
fn run_remote_queue_priority_probe(
    _command_path: &Path,
    _requested: &[QueuePriority],
    _submissions: usize,
) -> Result<()> {
    Err("--queue-priority-probe --command-socket requires Unix domain sockets".into())
}

/// Entry point for `--queue-priority-child`.
fn run_queue_priority_child_mode(path: std::ffi::OsString) -> Result<()> {
    #[cfg(unix)]
    {
        run_queue_priority_child(&path)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Err("--queue-priority-child requires Unix domain sockets".into())
    }
}

/// `--queue-priority-child <command-socket>`: the provider half of the remote
/// probe.
///
/// It owns a Vulkan device, serves the command channel, and once the owner
/// closes it reports the queue table it read, the selection sequence it
/// observed and the two provider-side observation surfaces it cross-checked.
/// This line is the provider-side half of the cross-process evidence.
#[cfg(unix)]
fn run_queue_priority_child(command_path: &std::ffi::OsStr) -> Result<()> {
    let executor = Arc::new(
        VulkanExecutor::new().map_err(|error| format!("create Vulkan executor: {error:?}"))?,
    );
    let observed = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&observed);
    executor.set_enqueue_probe_for_test(Arc::new(move |queue| {
        if let Ok(mut sequence) = sink.lock() {
            sequence.push(queue);
        }
    }));
    let provider = VulkanComputeProvider::with_executor(Arc::clone(&executor))
        .map_err(|error| format!("create Vulkan provider: {error:?}"))?
        .with_async_execution(true);
    let mut transport = command_unix::connect(command_path)?;
    serve_provider_unix(&provider, &mut transport)?;
    drop(transport);
    drop(provider);
    executor.clear_enqueue_probe_for_test();

    let installed = executor.queue_priorities();
    let counts = executor.queue_submission_counts();
    let sequence = observed
        .lock()
        .map_err(|_| "the enqueue probe sequence is poisoned")?
        .clone();
    if sequence.iter().any(|index| *index >= installed.len()) {
        return Err("the enqueue probe observed a queue outside the device".into());
    }
    let tiers: Vec<QueuePriority> = sequence.iter().map(|index| installed[*index]).collect();
    let share = |tier: QueuePriority| tiers.iter().filter(|seen| **seen == tier).count();
    println!(
        "queue_priority_child read={} queues={} submissions={} sequence={} index_counts={} \
         tier_counts=high={} default={} low={}",
        format_queue_tiers(&installed),
        installed.len(),
        sequence.len(),
        format_queue_sequence(&installed, &sequence),
        counts
            .iter()
            .map(|count| count.to_string())
            .collect::<Vec<_>>()
            .join(","),
        share(QueuePriority::High),
        share(QueuePriority::Default),
        share(QueuePriority::Low)
    );

    // The two provider-side observation surfaces have to agree, exactly as the
    // in-process probe requires.
    if counts.len() != installed.len() || counts.iter().sum::<usize>() != sequence.len() {
        return Err(format!(
            "queue_submission_counts() reports {counts:?} for {} submissions",
            sequence.len()
        )
        .into());
    }
    for (index, count) in counts.iter().enumerate() {
        let seen = sequence.iter().filter(|picked| **picked == index).count();
        if *count != seen {
            return Err(format!(
                "queue {index}: queue_submission_counts={count} enqueue_probe={seen}"
            )
            .into());
        }
    }
    println!(
        "PASS queue_priority_child queues={} submissions={} read={} probe_matches_counts=exact",
        installed.len(),
        sequence.len(),
        format_queue_tiers(&installed)
    );
    Ok(())
}

fn create_provider(
    backend: Backend,
    async_execution: bool,
    queue_priorities: Option<&[QueuePriority]>,
) -> Result<ProviderHandles> {
    match backend {
        Backend::Vulkan => {
            let executor = VulkanExecutor::new()
                .map_err(|error| format!("create Vulkan executor: {error:?}"))?;
            if let Some(requested) = queue_priorities {
                install_queue_priorities(&executor, requested)?;
            }
            let provider = Arc::new(
                VulkanComputeProvider::with_executor(Arc::clone(&executor))
                    .map_err(|error| format!("create Vulkan provider: {error:?}"))?
                    .with_async_execution(async_execution),
            );
            let name = provider.device_name().to_owned();
            // The render rail is a concrete-context entry point
            // (`register_render_pipeline` is not part of `PipelineProvider`), so
            // the handle is kept beside the trait object.
            Ok((
                Arc::clone(&provider) as Arc<dyn PipelineProvider>,
                name,
                CopyCounters::Vulkan(executor),
                Some(provider),
            ))
        }
        Backend::NativeMetalProvider => {
            #[cfg(target_os = "macos")]
            {
                let provider = Arc::new(
                    NativeMetalProvider::new()
                        .map_err(|error| format!("create native Metal provider: {error:?}"))?
                        .with_async_execution(async_execution),
                );
                let name = provider.device_name().to_owned();
                Ok((
                    Arc::clone(&provider) as Arc<dyn PipelineProvider>,
                    name,
                    CopyCounters::Native(provider),
                    None,
                ))
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
    /// Offscreen render cases (`research/docs/23` §1.2). A render case is not a
    /// compute case: its observable is one colour attachment's texels, so it
    /// lives in its own array and names the compute case whose pass declares
    /// the attachment view.
    #[serde(default)]
    render_cases: Vec<RenderCase>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    entry: String,
    grid: [u64; 3],
    local: [u64; 3],
    #[serde(default)]
    air_encoding: AirEncoding,
    air: Source,
    metal: Source,
    buffers: Vec<Buffer>,
    /// Sampled textures bound by every pass of the case (v11 and later).
    /// `research/docs/18` step 1.
    #[serde(default)]
    textures: Vec<Texture>,
    expected_writebacks: Vec<Writeback>,
    dispatches: Option<Vec<CaseDispatch>>,
    programs: Option<Vec<CaseProgram>>,
    command_buffers: Option<Vec<Vec<usize>>>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Texture {
    binding: u32,
    allocation: u64,
    view: u64,
    width: u64,
    height: u64,
    format: String,
    access: String,
    initial_hex: String,
}

/// One offscreen render case (`research/docs/23` §1.2, §5.1).
///
/// The shape is a whitelist rather than a per-case table: the first render
/// increment has exactly one render shape, so the shape *is* the review and a
/// fixture cannot widen it by renaming a case.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RenderCase {
    id: String,
    /// The compute case of this suite whose pass declares the attachment view.
    /// The render trace replays that pass, so the target resolves against a
    /// resource table the trace itself carries (`research/docs/23` §3.6).
    declaring_case: String,
    vertex_entry: String,
    fragment_entry: String,
    metal: Source,
    vertices: u64,
    viewport: [u64; 4],
    attachment: RenderAttachmentDefinition,
    expected_hex: String,
    /// The capture backends this case is executable on. A rail in this list has
    /// to report the case; a rail outside it has to omit it.
    capture_rails: Vec<String>,
}

/// The colour attachment a render case draws into. The fields mirror
/// `metal_api_core::provider::RenderAttachment`: identity, format, extent and
/// the load/store pair (`clear_hex` in memory order for a clear,
/// `initial_hex` for the previous contents).
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RenderAttachmentDefinition {
    allocation: u64,
    view: u64,
    format: String,
    width: u64,
    height: u64,
    load: String,
    store: String,
    clear_hex: Option<String>,
    initial_hex: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
enum AirEncoding {
    #[default]
    Text,
    Raw,
    Wrapped,
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

/// Device-buffer copy counters for one command buffer. A case that splits its
/// dispatch sequence across several command buffers submits once per group, so
/// the accumulated case counters cannot show which boundary copied what
/// (`research/docs/15` §5b).
#[derive(Serialize)]
struct GroupCounts {
    copy_in: u32,
    copy_out: u32,
}

#[derive(Serialize)]
struct CaseResult {
    id: String,
    completion: &'static str,
    writebacks: Vec<Writeback>,
    allocations: Vec<Allocation>,
    /// Device-buffer copy-in / copy-out operations for this case, summed over
    /// its submissions. One of each per touched allocation, not per view
    /// (`research/docs/15` §3.3). Absent from the Swift reference oracle,
    /// which is not a provider.
    copy_in: Option<u32>,
    copy_out: Option<u32>,
    /// Per-command-buffer counters, one entry per committed command buffer and
    /// in commit order. Recorded only for cases that split their sequence, so
    /// the flat totals above stay the sum of the groups
    /// (`research/docs/15` §5b).
    #[serde(skip_serializing_if = "Option::is_none")]
    group_counts: Option<Vec<GroupCounts>>,
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
    let mut api = None;
    let mut async_execution = false;
    let mut queue_priorities = None;
    let mut probe = false;
    let mut probe_submissions = None;
    let mut command_socket = None;
    while let Some(flag) = args.next() {
        if flag == "--help" {
            println!(
                "usage: provider-capture --suite conformance/suite.json [--output capture.json] \
                 [--backend vulkan|native-metal-provider] [--api trace|objects] [--async] \
                 [--queue-priorities low,default,high,...]\n\
                 usage: provider-capture --queue-priority-probe \
                 [--queue-priorities low,default,high,...] [--queue-priority-submissions N] \
                 [--command-socket <path>]\n\
                 usage: provider-capture --queue-priority-child <command-socket>"
            );
            return Ok(());
        }
        if flag == "--queue-priority-child" {
            let path = args
                .next()
                .ok_or("--queue-priority-child requires a command socket")?;
            return run_queue_priority_child_mode(path);
        }
        if flag == "--async" {
            async_execution = true;
            continue;
        }
        if flag == "--queue-priority-probe" {
            probe = true;
            continue;
        }
        if flag == "--command-socket" && command_socket.is_none() {
            command_socket = Some(PathBuf::from(
                args.next().ok_or("--command-socket requires a path")?,
            ));
            continue;
        }
        if flag == "--queue-priorities" && queue_priorities.is_none() {
            let value = args.next().ok_or("missing argument value")?;
            queue_priorities = Some(parse_queue_priorities(
                value
                    .to_str()
                    .ok_or("--queue-priorities must be valid UTF-8")?,
            )?);
            continue;
        }
        if flag == "--queue-priority-submissions" && probe_submissions.is_none() {
            let value = args.next().ok_or("missing argument value")?;
            probe_submissions = Some(
                value
                    .to_str()
                    .ok_or("--queue-priority-submissions must be valid UTF-8")?
                    .parse()
                    .map_err(|error| format!("--queue-priority-submissions: {error}"))?,
            );
            continue;
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
        if flag == "--api" && api.is_none() {
            api = Some(
                match args.next().as_deref().and_then(|value| value.to_str()) {
                    Some("trace") => EntryApi::Trace,
                    Some("objects") => EntryApi::Objects,
                    _ => return Err("--api requires trace or objects".into()),
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
    let api = api.unwrap_or(EntryApi::Trace);
    if async_execution && api != EntryApi::Objects {
        return Err("--async requires --api objects".into());
    }
    if probe {
        if suite_path.is_some() || output_path.is_some() || async_execution {
            return Err(
                "--queue-priority-probe runs its own scenario: it takes neither --suite, \
                 --output nor --async"
                    .into(),
            );
        }
        if backend == Backend::NativeMetalProvider {
            return Err("--queue-priority-probe requires the Vulkan backend".into());
        }
        if probe_submissions.is_some_and(|submissions| submissions == 0) {
            return Err("--queue-priority-submissions must be greater than zero".into());
        }
        let requested = queue_priorities.unwrap_or_else(default_queue_priorities);
        if requested.is_empty() {
            return Err("--queue-priorities must name at least one queue".into());
        }
        if let Some(path) = command_socket {
            // The provider lives in another process, so the marking and the
            // scenario both travel over the command channel.
            return run_remote_queue_priority_probe(
                &path,
                &requested,
                probe_submissions.unwrap_or(70),
            );
        }
        let executor =
            VulkanExecutor::new().map_err(|error| format!("create Vulkan executor: {error:?}"))?;
        install_queue_priorities(&executor, &requested)?;
        return run_queue_priority_probe(&executor, probe_submissions.unwrap_or(70));
    }
    if probe_submissions.is_some() {
        return Err("--queue-priority-submissions requires --queue-priority-probe".into());
    }
    if command_socket.is_some() {
        return Err("--command-socket requires --queue-priority-probe".into());
    }
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
            let source = match backend {
                Backend::Vulkan => {
                    let air = String::from_utf8(air)?;
                    match case.air_encoding {
                        AirEncoding::Text => ShaderSource::SanitizedLl(air),
                        AirEncoding::Raw => ShaderSource::BinaryAir(assemble_owned_air(&air)?),
                        AirEncoding::Wrapped => {
                            ShaderSource::BinaryAir(wrap_air_bitcode(&assemble_owned_air(&air)?)?)
                        }
                    }
                }
                Backend::NativeMetalProvider => {
                    ShaderSource::MetalSource(String::from_utf8(metal)?)
                }
            };
            sources.insert((program.entry, case.air_encoding), source);
        }
    }
    // A render case pins the one reviewed MSL module; the Vulkan rail executes
    // the matching SPIR-V pair pinned in code above, so only the module's
    // declared identity is verified here.
    for case in &suite.render_cases {
        verified_source(directory, &case.metal)?;
    }
    // Every rail has to agree about which cases it owns. A rail a render case's
    // marker names has to be able to report the case; the object-API rails carry
    // no render command encoder in this increment, so a suite that asks them for
    // one is refused instead of silently reporting fewer cases than the marker
    // requires.
    let render_rail = backend == Backend::Vulkan && api == EntryApi::Trace;
    for case in &suite.render_cases {
        if !render_rail
            && case
                .capture_rails
                .iter()
                .any(|rail| rail == backend.report_name(api))
        {
            return Err(format!(
                "render case {} is marked executable on {} but this rail cannot execute a \
                 render pass",
                case.id,
                backend.report_name(api)
            )
            .into());
        }
    }
    let identity = hex(&Sha256::digest(&raw));
    let (provider, device_name, counters, vulkan) =
        create_provider(backend, async_execution, queue_priorities.as_deref())?;
    let object_device =
        (api == EntryApi::Objects).then(|| objects::Device::new(Arc::clone(&provider)));
    let mut results = Vec::new();
    let mut pipelines = BTreeMap::new();
    let mut object_pipelines = BTreeMap::new();
    for ((entry, encoding), source) in sources {
        let request = PipelineCompileRequest {
            entry_name: entry.clone(),
            logical_digest: SemanticDigest::new(
                "suite-sha256-entry-v1",
                format!("{identity}:{entry}").into_bytes(),
            )?,
            source,
        };
        let pipeline = if let Some(device) = &object_device {
            let pipeline = device.compile_pipeline(request)?;
            let metadata = pipeline.metadata().clone();
            object_pipelines.insert((entry.clone(), encoding), pipeline);
            metadata
        } else {
            provider
                .compile(request)
                .map_err(|error| format!("compile {entry}: {error:?}"))?
        };
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
        pipelines.insert((entry, encoding), pipeline);
    }
    for (index, case) in suite.cases.iter().enumerate() {
        let programs = case_programs(case)
            .iter()
            .map(|program| pipelines[&(program.entry.clone(), case.air_encoding)].clone())
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
        let before = counters.read();
        let mut result = if let Some(device) = &object_device {
            let programs = case_programs(case)
                .iter()
                .map(|program| {
                    object_pipelines[&(program.entry.clone(), case.air_encoding)].clone()
                })
                .collect::<Vec<_>>();
            run_object_case(
                device,
                &programs,
                case,
                suite.guard_byte,
                async_execution,
                &mut || counters.read(),
            )?
        } else {
            run_case(
                provider.as_ref(),
                &programs,
                case,
                index as u64 + 1,
                suite.guard_byte,
                &mut || counters.read(),
            )?
        };
        let after = counters.read();
        result.copy_in = Some(u32::try_from(after.0 - before.0)?);
        result.copy_out = Some(u32::try_from(after.1 - before.1)?);
        results.push(result);
    }
    // Render cases run after the compute cases, on the one rail that owns a
    // render execution path (`research/docs/23` §6 Step 7). Every other rail
    // omits them, which is what the suite's `capture_rails` marker declares.
    let mut render_pipeline: Option<CompiledComputePipeline> = None;
    for (offset, case) in suite.render_cases.iter().enumerate() {
        if !render_rail {
            continue;
        }
        let vulkan = vulkan
            .as_ref()
            .ok_or("render cases require the Vulkan trace rail")?;
        let declaring = suite
            .cases
            .iter()
            .find(|declared| declared.id == case.declaring_case)
            .ok_or("render case declaring pass is not a case of this suite")?;
        let programs = case_programs(declaring)
            .iter()
            .map(|program| pipelines[&(program.entry.clone(), declaring.air_encoding)].clone())
            .collect::<Vec<_>>();
        let pipeline = match &render_pipeline {
            Some(pipeline) => pipeline.clone(),
            None => {
                let registered = vulkan
                    .register_render_pipeline(RenderPipelineRequest {
                        contract: RenderPipelineContract {
                            vertex_entry: RENDER_VERTEX_ENTRY.to_owned(),
                            fragment_entry: RENDER_FRAGMENT_ENTRY.to_owned(),
                            color_format: AttachmentFormat::Rgba8Unorm,
                            vertex_layout: VertexLayout::None,
                        },
                        vertex_spirv: RENDER_VERTEX_SPV.to_vec(),
                        fragment_spirv: RENDER_FRAGMENT_SPV.to_vec(),
                        logical_digest: SemanticDigest::new(
                            "suite-sha256-entry-v1",
                            format!("{identity}:offscreen_render_pipeline").into_bytes(),
                        )?,
                    })
                    .map_err(|error| format!("register render pipeline: {error:?}"))?;
                render_pipeline = Some(registered.clone());
                registered
            }
        };
        let before = counters.read();
        let mut result = run_render_case(
            provider.as_ref(),
            &programs,
            declaring,
            case,
            &pipeline,
            1000 + offset as u64,
            suite.guard_byte,
        )?;
        let after = counters.read();
        result.copy_in = Some(u32::try_from(after.0 - before.0)?);
        result.copy_out = Some(u32::try_from(after.1 - before.1)?);
        results.push(result);
    }
    if api == EntryApi::Trace {
        for pipeline in pipelines.values() {
            provider
                .release_pipeline(pipeline)
                .map_err(|error| format!("release pipeline: {error:?}"))?;
        }
    }
    if let Some(pipeline) = render_pipeline.as_ref() {
        vulkan
            .as_ref()
            .expect("only the Vulkan trace rail registers a render pipeline")
            .release_render_pipeline(pipeline)
            .map_err(|error| format!("release render pipeline: {error:?}"))?;
    }
    let capture = Capture {
        schema_version: 1,
        suite: suite.suite,
        suite_sha256: identity,
        backend: backend.report_name(api),
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
        (1, "compute-buffer-v8") => &[
            "subset_chain_two",
            "subset_chain_four",
            "subset_chain_eight",
        ],
        (1, "compute-buffer-v9") => &[
            "subset_chain_two",
            "subset_chain_four",
            "subset_chain_eight",
        ],
        (1, "compute-buffer-v10") => &["alias_disjoint_pair", "alias_disjoint_pair_reversed"],
        (1, "compute-buffer-v11") => &["sampled_texture_first_texel"],
        (1, "compute-buffer-v12") => &["texture_cell_local_4x4", "texture_cell_local_1x1"],
        (1, "compute-buffer-v13") => &["render_declaring_copy_word"],
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
    let encodings = suite
        .cases
        .iter()
        .map(|case| case.air_encoding)
        .collect::<BTreeSet<_>>();
    if suite.suite == "compute-buffer-v8" {
        if !encodings.contains(&AirEncoding::Raw) || !encodings.contains(&AirEncoding::Wrapped) {
            return Err("v8 suite must cover raw and wrapped binary AIR".into());
        }
    } else if encodings
        .iter()
        .any(|encoding| *encoding != AirEncoding::Text)
    {
        return Err("binary AIR encodings are only qualified by the v8 suite".into());
    }
    let mut ids = BTreeSet::new();
    // The view a render case draws into is the attachment's own allocation: the
    // render rail reads the image back directly, so the guard-byte discipline
    // that makes a compute case's offset mistake observable in the allocation
    // image does not apply to it (`research/docs/23` §5.2).
    let attachment_views = suite
        .render_cases
        .iter()
        .map(|case| (case.attachment.allocation, case.attachment.view))
        .collect::<BTreeSet<_>>();
    for case in &suite.cases {
        validate_case_programs(case)?;
        validate_case_dispatches(case)?;
        validate_case_command_buffers(&suite.suite, case)?;
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
        let mut allocations = BTreeMap::<u64, Vec<(u64, u64)>>::new();
        let mut views = BTreeSet::new();
        for (index, buffer) in case.buffers.iter().enumerate() {
            let end = buffer
                .offset
                .checked_add(buffer.length)
                .ok_or("view range overflow")?;
            let attachment_target = attachment_views.contains(&(buffer.allocation, buffer.view));
            if buffer.binding != buffers[index].0
                || buffer.access != buffers[index].1
                || buffer.length != buffers[index].2
                || buffer.allocation_size > MAX_BYTES as u64
                || end > buffer.allocation_size
                || (!attachment_target && (buffer.offset < 4 || buffer.allocation_size - end < 4))
                || !buffer.offset.is_multiple_of(4)
                || buffer.allocation == 0
                || buffer.view == 0
                || !views.insert(buffer.view)
            {
                return Err(format!("invalid buffer declaration in {}", case.id).into());
            }
            // Several buffers may name one allocation while their byte ranges
            // stay disjoint: that is the v10 ranged-alias shape. Overlapping
            // ranges would make the observed allocation image depend on write
            // order, so they are refused here exactly as provider admission
            // refuses them.
            let ranges = allocations.entry(buffer.allocation).or_default();
            if ranges
                .iter()
                .any(|(start, other_end)| buffer.offset < *other_end && *start < end)
            {
                return Err(format!(
                    "overlapping views of allocation {} in {}",
                    buffer.allocation, case.id
                )
                .into());
            }
            if unhex(&buffer.initial_hex)?.len() as u64 != buffer.length {
                return Err("initial data length differs from declared view length".into());
            }
            ranges.push((buffer.offset, end));
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
        // v11 texture section: every texture carries its full tightly packed
        // image, and each one names a distinct allocation the provider can
        // resolve.
        let mut texture_bindings = BTreeSet::new();
        let mut texture_allocations = BTreeSet::new();
        for texture in &case.textures {
            if !texture_bindings.insert(texture.binding) {
                return Err(format!("duplicate texture binding in {}", case.id).into());
            }
            if !texture_allocations.insert(texture.allocation) {
                return Err(format!("duplicate texture allocation in {}", case.id).into());
            }
            let expected = texture
                .width
                .checked_mul(texture.height)
                .and_then(|extent| extent.checked_mul(4))
                .ok_or("texture extent overflows")?;
            if unhex(&texture.initial_hex)?.len() as u64 != expected {
                return Err(format!(
                    "texture initial data length differs from declared extent in {}",
                    case.id
                )
                .into());
            }
        }
    }
    let mut render_ids = BTreeSet::new();
    for case in &suite.render_cases {
        validate_render_case(suite, case)?;
        if !render_ids.insert(&case.id) || ids.contains(&case.id) {
            return Err("duplicate render case identity".into());
        }
    }
    Ok(())
}

/// Validate one render case against the declaring case it draws into.
///
/// The rules mirror `conformance/compare.py`'s render plan and the Swift
/// oracle's `validateRenderCase`, including the two falsifiability rules: every
/// texel of the expectation has to be the fragment output, and the expectation
/// has to differ from the value the pass started from, so "the pass never ran"
/// cannot satisfy it.
fn validate_render_case(suite: &Suite, case: &RenderCase) -> Result<()> {
    let where_ = format!("render case {}", case.id);
    let attachment = &case.attachment;
    if attachment.format != "rgba8_unorm" {
        return Err(format!("{where_}: unsupported attachment format").into());
    }
    if attachment.width != 2 || attachment.height != 2 {
        return Err(
            format!("{where_}: the first render increment renders into a 2x2 attachment").into(),
        );
    }
    if attachment.allocation == 0 || attachment.view == 0 {
        return Err(format!("{where_}: zero attachment identity").into());
    }
    if attachment.store != "store" {
        return Err(format!("{where_}: a discarded attachment cannot be compared").into());
    }
    if case.vertices != 3 {
        return Err(format!("{where_}: expected the reviewed full-screen triangle").into());
    }
    if case.viewport != [0, 0, attachment.width, attachment.height] {
        return Err(format!("{where_}: the viewport must cover the attachment").into());
    }
    if case.vertex_entry != "render_fullscreen_triangle"
        || case.fragment_entry != "render_solid_rgba8"
    {
        return Err(format!("{where_}: unreviewed render pipeline identity").into());
    }
    let texels = unhex(&case.expected_hex)?;
    let extent = usize::try_from(
        attachment
            .width
            .checked_mul(attachment.height)
            .and_then(|texels| texels.checked_mul(4))
            .ok_or("attachment extent overflows")?,
    )?;
    if texels.len() != extent {
        return Err(format!("{where_}: expected texel bytes do not match the attachment").into());
    }
    let texel = &texels[..4];
    if texels.chunks_exact(4).any(|chunk| chunk != texel) {
        return Err(format!(
            "{where_}: every texel of the expectation has to be the fragment output"
        )
        .into());
    }
    match attachment.load.as_str() {
        "clear" => {
            let clear = unhex(
                attachment
                    .clear_hex
                    .as_deref()
                    .ok_or(format!("{where_}: a clear attachment needs clear_hex"))?,
            )?;
            if clear.len() != 4 {
                return Err(format!("{where_}: a clear colour is four bytes").into());
            }
            if attachment.initial_hex.is_some() {
                return Err(
                    format!("{where_}: a cleared attachment carries no initial bytes").into(),
                );
            }
            if clear == texel {
                return Err(format!("{where_}: the clear colour equals the expected texel").into());
            }
        }
        "load" => {
            let initial = unhex(attachment.initial_hex.as_deref().ok_or(format!(
                "{where_}: a loaded attachment needs its previous texels"
            ))?)?;
            if attachment.clear_hex.is_some() {
                return Err(
                    format!("{where_}: a loaded attachment carries no clear colour").into(),
                );
            }
            if initial.len() != extent {
                return Err(format!("{where_}: initial texels do not match the attachment").into());
            }
            if initial == texels {
                return Err(format!("{where_}: the initial texels equal the expectation").into());
            }
            // The rail has no attachment upload path in this increment: core
            // refuses `LoadOp::Load` at admission (`render_load_unsupported`),
            // so a case that asks for it cannot be reported honestly.
            return Err(format!(
                "{where_}: the Vulkan rail has no attachment upload path for LoadOp::Load yet"
            )
            .into());
        }
        other => return Err(format!("{where_}: unknown attachment load op {other:?}").into()),
    }
    let mut rails = BTreeSet::new();
    for rail in &case.capture_rails {
        if !RENDER_RAILS.contains(&rail.as_str()) {
            return Err(format!("{where_}: unknown capture rail {rail:?}").into());
        }
        if !rails.insert(rail) {
            return Err(format!("{where_}: duplicate capture rail {rail:?}").into());
        }
    }
    if rails.is_empty() {
        return Err(format!("{where_}: capture_rails cannot be empty").into());
    }
    // The attachment resolves against the declaring case's own table: one of
    // its declared views has to be the attachment, it has to be read-only (a
    // compute pass that wrote the view the render pass stores would make the
    // order inexpressible), and its byte range has to agree with the extent the
    // attachment restates.
    let declaring = suite
        .cases
        .iter()
        .find(|declared| declared.id == case.declaring_case)
        .ok_or(format!(
            "{where_}: unknown declaring case {}",
            case.declaring_case
        ))?;
    if dispatch_sequence(declaring).len() != 1
        || declaring.command_buffers.is_some()
        || declaring.programs.is_some()
    {
        return Err(format!(
            "{where_}: the declaring case must be one pass over its whole view pool"
        )
        .into());
    }
    let matches = declaring
        .buffers
        .iter()
        .filter(|buffer| {
            buffer.allocation == attachment.allocation && buffer.view == attachment.view
        })
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!(
            "{where_}: the declaring case has to declare exactly the attachment view"
        )
        .into());
    }
    let declared = matches[0];
    if declared.access != "read" {
        return Err(
            format!("{where_}: the declaring pass must only read the attachment view").into(),
        );
    }
    if declared.length != u64::try_from(texels.len())? {
        return Err(
            format!("{where_}: attachment extent disagrees with the declaring view").into(),
        );
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

fn dispatch_sequence(case: &Case) -> Vec<CaseDispatch> {
    case.dispatches.clone().unwrap_or_else(|| {
        vec![CaseDispatch {
            grid: case.grid,
            local: case.local,
            bindings: None,
            program: None,
        }]
    })
}

/// Dispatch indices per command buffer. Legacy fixtures submit one command
/// buffer; the v9 suite splits the same sequence across several.
fn case_command_buffers(case: &Case) -> Vec<Vec<usize>> {
    case.command_buffers
        .clone()
        .unwrap_or_else(|| vec![(0..dispatch_sequence(case).len()).collect()])
}

fn validate_case_command_buffers(suite: &str, case: &Case) -> Result<()> {
    let dispatches = dispatch_sequence(case);
    let Some(groups) = &case.command_buffers else {
        if suite == "compute-buffer-v9" {
            return Err("v9 fixture requires command buffer groups".into());
        }
        return Ok(());
    };
    if suite != "compute-buffer-v9" {
        return Err("command buffer groups are only qualified by the v9 suite".into());
    }
    if !(2..=4).contains(&groups.len()) {
        return Err("v9 fixture needs two to four command buffers".into());
    }
    let mut expected = 0usize;
    for group in groups {
        if group.is_empty() {
            return Err("command buffer group cannot be empty".into());
        }
        for index in group {
            if *index != expected {
                return Err("command buffer groups must partition the dispatch order".into());
            }
            expected += 1;
        }
    }
    if expected != dispatches.len() {
        return Err("command buffer groups must partition the dispatch order".into());
    }
    Ok(())
}

/// Command buffers may each report writes for the same view; the capture
/// reports the final per-view landing, so later writes overlay earlier ones
/// and every written view must end up fully covered.
fn merge_writebacks(
    case: &Case,
    entries: impl IntoIterator<Item = (u64, u64, u64, Vec<u8>)>,
) -> Result<Vec<Writeback>> {
    let mut views = BTreeMap::new();
    for buffer in &case.buffers {
        let length = usize::try_from(buffer.length)?;
        views.insert(
            (buffer.allocation, buffer.view),
            (buffer.offset, vec![0_u8; length], vec![false; length]),
        );
    }
    for (allocation, view, offset, bytes) in entries {
        let (view_offset, data, covered) = views
            .get_mut(&(allocation, view))
            .ok_or("writeback references an unknown view")?;
        let start = usize::try_from(
            offset
                .checked_sub(*view_offset)
                .ok_or("writeback starts before its view")?,
        )?;
        let end = start
            .checked_add(bytes.len())
            .ok_or("writeback range overflow")?;
        data.get_mut(start..end)
            .ok_or("writeback exceeds its view")?
            .copy_from_slice(&bytes);
        covered[start..end].fill(true);
    }
    let mut writebacks = Vec::new();
    for ((allocation, view), (offset, data, covered)) in views {
        if covered.iter().all(|value| *value) {
            writebacks.push(Writeback {
                allocation,
                view,
                offset,
                bytes_hex: hex(&data),
            });
        } else if covered.iter().any(|value| *value) {
            return Err(format!("writebacks do not cover view {view} exactly").into());
        }
    }
    Ok(writebacks)
}

fn validate_program(program: &CaseProgram) -> Result<()> {
    let (air_path, air_hash, metal_path, metal_hash) = match program.entry.as_str() {
        "read_texture_2d" => (
            "../examples/metal-smoke/shaders/kernel_read_texture_2d.ll",
            "3e969b61d3149bc9351f44c56de6fb85a403557cbcbee7240602539ca794c8df",
            "shaders/read_texture_2d.metal",
            "da21ca69d76018f2911aaf6867f517fca8e41b20d531b6b43df30931563499ee",
        ),
        "read_texture_2d_cell" => (
            "../examples/metal-smoke/shaders/kernel_read_texture_2d_cell.ll",
            "80fe6866bac049de9c1c2b33d9f15a3a133b68c321dfdb16721c991f8dfc23c9",
            "shaders/read_texture_2d_cell.metal",
            "6517da4354381bb46706ec3395d3e449ff08499df37c0c1f2a620a0c04161237",
        ),
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
        // v11: the texture case binds a sampled texture plus one write-only
        // output buffer; textures are validated by the texture section.
        "sampled_texture_first_texel" => (
            "read_texture_2d",
            [1, 1, 1],
            [1, 1, 1],
            &[(0, "write", 64)][..],
        ),
        // v12: the same 4x4 grid runs once as a single 4x4 group and once as
        // sixteen 1x1 groups. Both forms read texel(x, y), so the two shapes
        // must agree on [100..115] for a 4x4 R32Uint image holding 0..15.
        "texture_cell_local_4x4" => (
            "read_texture_2d_cell",
            [4, 4, 1],
            [4, 4, 1],
            &[(0, "write", 64)][..],
        ),
        "texture_cell_local_1x1" => (
            "read_texture_2d_cell",
            [4, 4, 1],
            [1, 1, 1],
            &[(0, "write", 64)][..],
        ),
        // v13: the declaring pass of the render suite reads the attachment's
        // own allocation (the whole 2x2x4 image) and copies its first word into
        // a second allocation, so the render submission also proves the
        // declaring pass really read the attachment view. The attachment
        // allocation carries no guard bytes: it is the attachment.
        "render_declaring_copy_word" => (
            "copy_word",
            [1, 1, 1],
            [1, 1, 1],
            &[(0, "read", 16), (1, "write", 4)][..],
        ),
        "copy_word" | "copy_seed_a" | "copy_seed_b" | "copy_pingpong" => copy,
        // v10: two disjoint views of one allocation. The reversed pair binds
        // the source above the destination so an offset mix-up cannot pass.
        "alias_disjoint_pair" | "alias_disjoint_pair_reversed" => copy,
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
    textures: &[TextureView],
    dispatches: &[CaseDispatch],
) -> Result<ComputeTrace> {
    let passes = dispatches
        .iter()
        .map(|dispatch| -> Result<ComputePass> {
            let selected = &programs[dispatch.program.unwrap_or(0)];
            let expected = selected_slots(case, dispatch);
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
                textures: textures.to_vec(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    // The provider rejects unused pipeline metadata, so each command buffer
    // carries only the programs its own passes reference.
    let mut pipelines: Vec<CompiledComputePipeline> = Vec::new();
    for pass in &passes {
        if pipelines
            .iter()
            .any(|program| program.pipeline_id == pass.pipeline)
        {
            continue;
        }
        let program = programs
            .iter()
            .find(|program| program.pipeline_id == pass.pipeline)
            .ok_or("pass pipeline is not part of the compiled program table")?;
        pipelines.push(program.clone());
    }
    Ok(ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch,
        operation_id: OperationId::new(operation),
        pipelines,
        encoder_dispatch_type: DispatchType::Serial,
        passes: passes.into_iter().map(TracePass::Compute).collect(),
        completion_policy: CompletionPolicy::HostReadback,
    })
}

/// Execute one render case on the Vulkan trace rail.
///
/// The trace is the declaring case's own pass followed by the render pass, i.e.
/// the shape `research/docs/23` §3.6 admits: the attachment references a view
/// the trace declares, the declaring pass only *reads* it, and the render rail
/// executes after the compute sequence. The attachment's bytes leave through
/// the existing writeback channel — the render rail pushes one
/// `BufferWriteback` for the view the trace declares — so this reports the
/// attachment's allocation image and that one writeback and nothing else. The
/// declaring pass's own landing belongs to the declaring case, which is what
/// keeps an attachment observation from being confusable with a buffer
/// writeback.
fn run_render_case(
    provider: &dyn PipelineProvider,
    programs: &[CompiledComputePipeline],
    declaring: &Case,
    case: &RenderCase,
    render_pipeline: &CompiledComputePipeline,
    operation: u64,
    guard: u8,
) -> Result<CaseResult> {
    let attachment = &case.attachment;
    // The declaring pass's own resource table: one backing image and one
    // `AllocationRecord` per allocation (`docs/23` §4.1).
    let mut allocations: Vec<(u64, Vec<u8>)> = Vec::new();
    let mut recorded = BTreeSet::new();
    let mut resources = ResourceTableSnapshot::new();
    for buffer in &declaring.buffers {
        let initial = unhex(&buffer.initial_hex)?;
        let start = usize::try_from(buffer.offset)?;
        let position = match allocations
            .iter()
            .position(|(allocation, _)| *allocation == buffer.allocation)
        {
            Some(position) => position,
            None => {
                allocations.push((
                    buffer.allocation,
                    vec![guard; usize::try_from(buffer.allocation_size)?],
                ));
                allocations.len() - 1
            }
        };
        if recorded.insert(buffer.allocation) {
            resources.insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(buffer.allocation),
                owner_epoch: provider.device_epoch(),
                size: buffer.allocation_size,
            })?;
        }
        allocations[position].1[start..start + initial.len()].copy_from_slice(&initial);
    }
    let views = declaring
        .buffers
        .iter()
        .map(|buffer| {
            let access = match buffer.access.as_str() {
                "read" => BufferAccess::Read,
                "write" => BufferAccess::Write,
                "read_write" => BufferAccess::ReadWrite,
                _ => return Err("unsupported access".into()),
            };
            let (_, backing) = allocations
                .iter()
                .find(|(allocation, _)| *allocation == buffer.allocation)
                .ok_or("unknown fixture allocation")?;
            let start = usize::try_from(buffer.offset)?;
            let end = start + usize::try_from(buffer.length)?;
            Ok(BufferView {
                view_id: ViewId::new(buffer.view),
                metal_binding: buffer.binding,
                allocation_id: AllocationId::new(buffer.allocation),
                offset: buffer.offset,
                length: buffer.length,
                access,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(backing[start..end].to_vec()),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let declared = views
        .iter()
        .find(|view| {
            view.view_id == ViewId::new(attachment.view)
                && view.allocation_id == AllocationId::new(attachment.allocation)
        })
        .ok_or("the declaring pass does not declare the attachment view")?
        .clone();

    let mut trace = case_trace(
        provider.device_epoch(),
        programs,
        declaring,
        operation,
        &views,
        &[],
        &dispatch_sequence(declaring),
    )?;
    let clear = unhex(
        attachment
            .clear_hex
            .as_deref()
            .ok_or("a clear attachment needs clear_hex")?,
    )?;
    trace.pipelines.push(render_pipeline.clone());
    trace.passes.push(TracePass::Render(RenderPassDescriptor {
        pipeline: render_pipeline.pipeline_id,
        color_attachments: vec![RenderAttachment {
            view_id: ViewId::new(attachment.view),
            allocation_id: AllocationId::new(attachment.allocation),
            format: AttachmentFormat::Rgba8Unorm,
            width: attachment.width,
            height: attachment.height,
            load: LoadOp::Clear(ClearColor::new(
                clear
                    .try_into()
                    .map_err(|_| -> Box<dyn Error> { "a clear colour is four bytes".into() })?,
            )),
            store: StoreOp::Store,
        }],
        viewport: [
            u32::try_from(case.viewport[0])?,
            u32::try_from(case.viewport[1])?,
            u32::try_from(case.viewport[2])?,
            u32::try_from(case.viewport[3])?,
        ],
        vertices: u32::try_from(case.vertices)?,
    }));

    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources.clone())
        .map_err(|error| format!("admit {}: {error:?}", case.id))?;
    let output = provider
        .submit(admitted)
        .map_err(|error| format!("submit {}: {error:?}", case.id))?;
    output.validate_for_trace(&trace)?;
    let CompletionDisposition::CompletedVisible { token } = output.completion else {
        return Err("render capture requires completed visible results".into());
    };
    if provider
        .wait(token, Duration::ZERO)
        .map_err(|error| format!("wait: {error:?}"))?
        != output.completion
    {
        return Err("provider completion observation changed".into());
    }
    let mut landed = None;
    for write in output.writebacks {
        let (_, backing) = allocations
            .iter_mut()
            .find(|(id, _)| *id == write.allocation_id.get())
            .ok_or("unknown writeback allocation")?;
        let start = usize::try_from(write.offset)?;
        backing[start..start + write.bytes.len()].copy_from_slice(&write.bytes);
        if write.view_id == declared.view_id && write.allocation_id == declared.allocation_id {
            landed = Some(write);
        }
    }
    provider
        .release_completion(token)
        .map_err(|error| format!("release completion: {error:?}"))?;
    let landed = landed.ok_or("the render rail landed no attachment writeback")?;
    // The observation is the attachment's own range: a rail that landed a
    // different range cannot be reported as this case's texels.
    if landed.offset != declared.offset || landed.bytes.len() as u64 != declared.length {
        return Err(format!(
            "render case {}: the attachment writeback covers {}..{} instead of {}..{}",
            case.id,
            landed.offset,
            landed.offset + landed.bytes.len() as u64,
            declared.offset,
            declared.offset + declared.length
        )
        .into());
    }
    let image = allocations
        .iter()
        .find(|(id, _)| *id == attachment.allocation)
        .ok_or("the attachment allocation is missing")?
        .1
        .clone();
    eprintln!(
        "render case completed: {} attachment={} bytes={}",
        case.id,
        hex(&landed.bytes),
        landed.bytes.len()
    );
    Ok(CaseResult {
        id: case.id.clone(),
        completion: "CompletedVisible",
        writebacks: vec![Writeback {
            allocation: attachment.allocation,
            view: attachment.view,
            offset: landed.offset,
            bytes_hex: hex(&landed.bytes),
        }],
        allocations: vec![Allocation {
            allocation: attachment.allocation,
            bytes_hex: hex(&image),
        }],
        copy_in: None,
        copy_out: None,
        group_counts: None,
    })
}

fn run_object_case(
    device: &objects::Device,
    programs: &[objects::Pipeline],
    case: &Case,
    guard: u8,
    async_execution: bool,
    counters: &mut dyn FnMut() -> (usize, usize),
) -> Result<CaseResult> {
    // Fixture IDs are report labels only. The object API creates and validates
    // its own allocation/view identities before they are mapped back here.
    let mut resources = BTreeMap::new();
    let mut report_ids = BTreeMap::new();
    // One device Buffer per allocation. A v10 fixture binds several disjoint
    // views of the same allocation, and sharing the object is what makes the
    // provider exercise ranged aliasing instead of seeing unrelated
    // allocations. Guard bytes outside every view stay in the shared image so
    // an offset or extent mistake stays observable.
    let mut images = BTreeMap::<u64, Vec<u8>>::new();
    for definition in &case.buffers {
        let size = usize::try_from(definition.allocation_size)?;
        let offset = usize::try_from(definition.offset)?;
        let bytes = unhex(&definition.initial_hex)?;
        let image = images
            .entry(definition.allocation)
            .or_insert_with(|| vec![guard; size]);
        image[offset..offset + bytes.len()].copy_from_slice(&bytes);
    }
    let mut allocation_buffers = BTreeMap::<u64, objects::Buffer>::new();
    for definition in &case.buffers {
        let buffer = match allocation_buffers.get(&definition.allocation) {
            Some(existing) => existing.clone(),
            None => {
                let image = images
                    .remove(&definition.allocation)
                    .ok_or("missing allocation image")?;
                let created = device.new_buffer_with_bytes(image)?;
                allocation_buffers.insert(definition.allocation, created.clone());
                created
            }
        };
        let view = buffer.view(
            usize::try_from(definition.offset)?,
            usize::try_from(definition.length)?,
        )?;
        report_ids.insert(
            (view.allocation_id(), view.view_id()),
            (definition.allocation, definition.view),
        );
        resources.insert(definition.view, (buffer, view));
    }
    let queue = device.new_command_queue();
    let dispatches = dispatch_sequence(case);
    let groups = case_command_buffers(case);
    let narrow = |dimensions: [u64; 3]| -> Result<Size> {
        Ok(Size::new(
            u32::try_from(dimensions[0])?,
            u32::try_from(dimensions[1])?,
            u32::try_from(dimensions[2])?,
        )?)
    };
    // v11: sampled textures become object-API Texture handles, bound by the
    // same Metal argument index as the AIR fixture declares.
    let mut object_textures = BTreeMap::new();
    for texture in &case.textures {
        let initial = unhex(&texture.initial_hex)?;
        let created = device.new_texture_with_bytes(
            TextureFormat::R32Uint,
            texture.width,
            texture.height,
            initial,
        )?;
        object_textures.insert(texture.binding, created);
    }
    let mut reported = Vec::new();
    let mut group_counts = Vec::with_capacity(groups.len());
    // Each command buffer commits and completes before the next one records,
    // which matches Metal's serial queue boundary and re-snapshots the landed
    // bytes for the following command.
    for group in &groups {
        let before = counters();
        let command = queue.command_buffer();
        // Several dispatches on one encoder exercise snapshot-at-dispatch behavior,
        // including changed pipelines, binding tables and later first use.
        let mut encoder = command.compute_command_encoder()?;
        for index in group {
            let dispatch = dispatches
                .get(*index)
                .ok_or("command buffer dispatch index out of range")?;
            encoder.clear_buffers()?;
            encoder.clear_textures()?;
            encoder.set_compute_pipeline_state(&programs[dispatch.program.unwrap_or(0)])?;
            for (binding, texture) in &object_textures {
                encoder.set_texture(*binding, texture)?;
            }
            let slots = selected_slots(case, dispatch);
            let views = dispatch
                .bindings
                .clone()
                .unwrap_or_else(|| case.buffers.iter().map(|buffer| buffer.view).collect());
            for (slot, view) in slots.iter().zip(views) {
                let (_, view) = resources.get(&view).ok_or("unknown object fixture view")?;
                encoder.set_buffer(slot.binding, view)?;
            }
            encoder.dispatch_threads(narrow(dispatch.grid)?, narrow(dispatch.local)?)?;
        }
        encoder.end_encoding()?;
        command.commit()?;
        if async_execution {
            if command.status()? != metal_api_core::CommandBufferStatus::Committed {
                return Err("async object commit did not leave the command pending".into());
            }
            if !matches!(
                command.submission()?.completion,
                CompletionDisposition::Submitted { .. }
            ) {
                return Err("async object commit did not return a submitted token".into());
            }
        }
        command.wait_until_completed()?;
        if command.status()? != metal_api_core::CommandBufferStatus::Completed {
            return Err("object command did not reach Completed".into());
        }
        let output = command.submission()?;
        if !matches!(
            output.completion,
            CompletionDisposition::CompletedVisible { .. }
        ) {
            return Err("object capture requires completed visible results".into());
        }
        for write in output.writebacks {
            let &(allocation, view) = report_ids
                .get(&(write.allocation_id, write.view_id))
                .ok_or("unknown object writeback identity")?;
            reported.push((allocation, view, write.offset, write.bytes));
        }
        let after = counters();
        group_counts.push(GroupCounts {
            copy_in: u32::try_from(after.0 - before.0)?,
            copy_out: u32::try_from(after.1 - before.1)?,
        });
    }
    let writebacks = merge_writebacks(case, reported)?;
    let mut allocations = Vec::new();
    for (allocation, buffer) in &allocation_buffers {
        // Observe the object's actual host landing, rather than replaying the
        // returned writebacks into a second synthetic allocation. Exactly one
        // entry per allocation: several views may share it.
        allocations.push(Allocation {
            allocation: *allocation,
            bytes_hex: hex(&buffer.read()?),
        });
    }
    allocations.sort_by_key(|allocation| allocation.allocation);
    eprintln!(
        "objects command completed: {} command_buffers={} passes={}",
        case.id,
        groups.len(),
        dispatches.len()
    );
    Ok(CaseResult {
        id: case.id.clone(),
        completion: "CompletedVisible",
        writebacks,
        allocations,
        copy_in: None,
        copy_out: None,
        group_counts: case.command_buffers.as_ref().map(|_| group_counts),
    })
}

fn run_case(
    provider: &dyn PipelineProvider,
    programs: &[CompiledComputePipeline],
    case: &Case,
    operation: u64,
    guard: u8,
    counters: &mut dyn FnMut() -> (usize, usize),
) -> Result<CaseResult> {
    let mut resources = ResourceTableSnapshot::new();
    // One backing image and one AllocationRecord per allocation. A v10 fixture
    // binds two disjoint views of the same allocation, so every view's initial
    // bytes land in that allocation's single image and the trace carries one
    // record with two view ranges.
    let mut allocations: Vec<(u64, Vec<u8>)> = Vec::new();
    let mut recorded = BTreeSet::new();
    for buffer in &case.buffers {
        let initial = unhex(&buffer.initial_hex)?;
        let start = usize::try_from(buffer.offset)?;
        let position = match allocations
            .iter()
            .position(|(allocation, _)| *allocation == buffer.allocation)
        {
            Some(position) => position,
            None => {
                allocations.push((
                    buffer.allocation,
                    vec![guard; usize::try_from(buffer.allocation_size)?],
                ));
                allocations.len() - 1
            }
        };
        if recorded.insert(buffer.allocation) {
            resources.insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(buffer.allocation),
                owner_epoch: provider.device_epoch(),
                size: buffer.allocation_size,
            })?;
        }
        allocations[position].1[start..start + initial.len()].copy_from_slice(&initial);
    }
    // v11: sampled textures are their own allocations; the provider uploads
    // them once per submission (`research/docs/18` step 1).
    let mut case_textures = Vec::with_capacity(case.textures.len());
    for texture in &case.textures {
        let initial = unhex(&texture.initial_hex)?;
        if recorded.insert(texture.allocation) {
            resources.insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(texture.allocation),
                owner_epoch: provider.device_epoch(),
                size: u64::try_from(initial.len())?,
            })?;
        }
        let access = match texture.access.as_str() {
            "sampled" => TextureAccess::Sampled,
            "storage" => TextureAccess::Storage,
            _ => return Err("unsupported texture access".into()),
        };
        let format = match texture.format.as_str() {
            "r32_uint" => TextureFormat::R32Uint,
            _ => return Err("unsupported texture format".into()),
        };
        case_textures.push(TextureView {
            view_id: ViewId::new(texture.view),
            metal_binding: texture.binding,
            allocation_id: AllocationId::new(texture.allocation),
            texture_type: TextureType::D2,
            format,
            width: texture.width,
            height: texture.height,
            depth: 1,
            array_length: 1,
            sample_count: 1,
            access,
            source: TextureSource::OwnedBytes(initial),
        });
    }
    // Every command buffer snapshots the bytes landed so far, so a later
    // command reads what an earlier command wrote.
    let case_views = |allocations: &[(u64, Vec<u8>)]| -> Result<Vec<BufferView>> {
        case.buffers
            .iter()
            .map(|buffer| {
                let access = match buffer.access.as_str() {
                    "read" => BufferAccess::Read,
                    "write" => BufferAccess::Write,
                    "read_write" => BufferAccess::ReadWrite,
                    _ => return Err("unsupported access".into()),
                };
                let (_, backing) = allocations
                    .iter()
                    .find(|(allocation, _)| *allocation == buffer.allocation)
                    .ok_or("unknown fixture allocation")?;
                let start = usize::try_from(buffer.offset)?;
                let end = start + usize::try_from(buffer.length)?;
                Ok(BufferView {
                    view_id: ViewId::new(buffer.view),
                    metal_binding: buffer.binding,
                    allocation_id: AllocationId::new(buffer.allocation),
                    offset: buffer.offset,
                    length: buffer.length,
                    access,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(backing[start..end].to_vec()),
                })
            })
            .collect()
    };
    let dispatches = dispatch_sequence(case);
    let groups = case_command_buffers(case);
    let initial_views = case_views(&allocations)?;
    // Refusal guards run on the complete sequence before any submission.
    let guard_trace = case_trace(
        provider.device_epoch(),
        programs,
        case,
        operation,
        &initial_views,
        &case_textures,
        &dispatches,
    )?;
    if case.entry == "transform_3d" {
        let mut short = guard_trace.clone();
        let shortened = short.passes[0]
            .as_compute()
            .ok_or("guard trace pass is not a compute pass")?
            .buffers[0]
            .view_id;
        for pass in short
            .passes
            .iter_mut()
            .filter_map(TracePass::as_compute_mut)
        {
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
        let mut forged = guard_trace.clone();
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
        let mut unknown = guard_trace.clone();
        let original_id = unknown.pipelines[1].pipeline_id;
        let missing = metal_api_core::provider::PipelineId::new(u64::MAX);
        unknown.pipelines[1].pipeline_id = missing;
        for pass in unknown
            .passes
            .iter_mut()
            .filter_map(TracePass::as_compute_mut)
        {
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
    let mut reported = Vec::new();
    let mut group_counts = Vec::with_capacity(groups.len());
    for (index, group) in groups.iter().enumerate() {
        let before = counters();
        let views = case_views(&allocations)?;
        let selected = group
            .iter()
            .map(|position| {
                dispatches
                    .get(*position)
                    .cloned()
                    .ok_or_else(|| -> Box<dyn std::error::Error> {
                        "command buffer dispatch index out of range".into()
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let trace_operation = if groups.len() == 1 {
            operation
        } else {
            operation * 100 + index as u64 + 1
        };
        let trace = case_trace(
            provider.device_epoch(),
            programs,
            case,
            trace_operation,
            &views,
            &case_textures,
            &selected,
        )?;
        let admitted = provider
            .capabilities()
            .validate_trace(trace.clone(), resources.clone())
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
        for write in output.writebacks {
            let (_, backing) = allocations
                .iter_mut()
                .find(|(id, _)| *id == write.allocation_id.get())
                .ok_or("unknown writeback allocation")?;
            let start = usize::try_from(write.offset)?;
            backing[start..start + write.bytes.len()].copy_from_slice(&write.bytes);
            reported.push((
                write.allocation_id.get(),
                write.view_id.get(),
                write.offset,
                write.bytes,
            ));
        }
        provider
            .release_completion(token)
            .map_err(|error| format!("release completion: {error:?}"))?;
        let after = counters();
        group_counts.push(GroupCounts {
            copy_in: u32::try_from(after.0 - before.0)?,
            copy_out: u32::try_from(after.1 - before.1)?,
        });
    }
    let writebacks = merge_writebacks(case, reported)?;
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
        copy_in: None,
        copy_out: None,
        group_counts: case.command_buffers.as_ref().map(|_| group_counts),
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
    fn v9_splits_the_reviewed_sequence_across_command_buffers() {
        let load = || {
            serde_json::from_str::<Suite>(include_str!("../../../../conformance/suite-v9.json"))
                .unwrap()
        };
        let s = load();
        validate_suite(&s).unwrap();
        for case in &s.cases {
            let groups = case_command_buffers(case);
            assert_eq!(
                groups.len(),
                if case.id == "subset_chain_four" {
                    2
                } else {
                    groups.len()
                }
            );
            let flattened: Vec<usize> = groups.iter().flatten().copied().collect();
            assert_eq!(
                flattened,
                (0..dispatch_sequence(case).len()).collect::<Vec<_>>()
            );
            assert!(groups.iter().all(|group| !group.is_empty()));
        }
        let mut s = load();
        s.cases[0].command_buffers = None;
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[0].command_buffers = Some(vec![vec![1], vec![0]]);
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[0].command_buffers = Some(vec![vec![0, 1], vec![]]);
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[0].command_buffers = Some(vec![vec![0], vec![1], vec![2]]);
        assert!(validate_suite(&s).is_err());
        let mut legacy = suite();
        legacy.cases[0].command_buffers = Some(vec![vec![0]]);
        assert!(validate_suite(&legacy).is_err());
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
            let trace = case_trace(
                DeviceEpoch::new(1),
                &programs,
                case,
                1,
                &views,
                &[],
                &dispatch_sequence(case),
            )
            .unwrap();
            let resources = trace.serial_resources().unwrap();
            assert_eq!(resources.len(), case.buffers.len());
            assert_eq!(
                trace.passes[0]
                    .as_compute()
                    .expect("capture pass is a compute pass")
                    .buffers
                    .len(),
                3
            );
            assert_eq!(
                trace.passes[1]
                    .as_compute()
                    .expect("capture pass is a compute pass")
                    .buffers
                    .len(),
                2
            );
            assert_eq!(
                trace.passes[1]
                    .as_compute()
                    .expect("capture pass is a compute pass")
                    .buffers
                    .iter()
                    .map(|view| (view.metal_binding, view.view_id.get()))
                    .collect::<Vec<_>>(),
                [(4, 400), (9, 430)]
            );
            assert!(!trace.passes[1]
                .as_compute()
                .expect("capture pass is a compute pass")
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
            assert!(case_trace(
                DeviceEpoch::new(1),
                &programs,
                case,
                1,
                &views,
                &[],
                &dispatch_sequence(case)
            )
            .is_err());
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
