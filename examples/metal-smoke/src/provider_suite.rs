//! Live checks of canonical provider submission against the snapshot executor.

use super::{
    assemble_owned_air, execute_copy_word, execute_indexed_boundary_dispatch,
    indexed_boundary_golden, wrap_air_bitcode,
};
use metal_api_core::completion::wire::{
    CompletionMessage, CompletionOutbox, CompletionSink, CompletionUpdate, MirrorOutcome,
};
use metal_api_core::provider::{
    AllocationId, AllocationRecord, BorrowedLease, BufferAccess, BufferLease, BufferSource,
    BufferView, CompletionDisposition, CompletionPolicy, CompletionToken, ComputePass,
    ComputeProvider, ComputeTrace, DeviceEpoch, Dispatch, DispatchKind, DispatchType,
    FootprintProof, LeaseId, LeaseImporter, LeaseLedger, LeaseObservation, LeaseReservation,
    NoCopyLeaseImporter, OperationId, PipelineCompileRequest, PipelineProvider, ProviderError,
    ProviderHealth, ProviderSubmission, ResourceTableSnapshot, SemanticDigest, ShaderSource,
    StagedLease, StorageMode, SubmissionId, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{Device, Library};
use metal_api_ipc::command::{serve_provider_unix, unix as command_unix, RemoteProvider};
use metal_api_ipc::receiver::CompletionReceiver;
use metal_api_ipc::sender::spawn_writer;
use metal_api_ipc::{shared, unix};
use metal_api_vulkan::{CompiledComputePipeline, VulkanComputeProvider, VulkanExecutor};
use std::error::Error;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const BORROWED_SHARED_LEASE_ID: u64 = 99;
const BORROWED_SHARED_ALLOCATION_ID: u64 = 298;
const BORROWED_SHARED_LENGTH: u64 = 64;
const BORROWED_SHARED_SIZE: usize = 4096;
const BORROWED_SHARED_OWNER_WORD: u32 = 0xaaaa_aaaa;
const BORROWED_SHARED_GPU_WORD: u32 = 0x1234_5678;

#[derive(Default)]
struct RecordingSink {
    messages: Mutex<Vec<CompletionMessage>>,
}

impl CompletionSink for RecordingSink {
    fn deliver(&self, message: CompletionMessage) {
        self.messages
            .lock()
            .expect("recording completion sink")
            .push(message);
    }
}

/// Exercise provider admission, GPU execution, writeback identity, and completion.
/// Both paths share one Vulkan executor but compile and submit independently.
pub fn run_provider_suite(executor: Arc<VulkanExecutor>) -> Result<(), Box<dyn Error>> {
    println!("Metal API provider: standalone Vulkan");
    println!("Metal API Vulkan device: {}", executor.device_name());
    let provider =
        VulkanComputeProvider::with_executor(executor.clone()).map_err(provider_error)?;
    let peer = VulkanComputeProvider::with_executor(executor.clone()).map_err(provider_error)?;
    let device = Device::new(Arc::clone(&executor) as Arc<dyn metal_api_core::ComputeExecutor>);
    let source = include_str!("../shaders/kernel_copy_word.ll");
    run_copy(
        &provider,
        &device,
        device.new_library_with_air(source)?,
        "textual",
        1,
    )?;
    let raw = assemble_owned_air(source)?;
    let wrapped = wrap_air_bitcode(&raw)?;
    for (index, (encoding, air)) in [("raw", raw), ("wrapped", wrapped)].into_iter().enumerate() {
        run_copy(
            &provider,
            &device,
            device.new_library_with_binary_air(air)?,
            encoding,
            index as u64 + 2,
        )?;
    }
    run_indexed_and_refusals(&provider, &peer, &device)?;
    run_timeout_reclamation(Arc::clone(&executor))?;
    run_cancellation(Arc::clone(&executor))?;
    run_completion_ipc(Arc::clone(&executor))?;
    run_completion_ipc_process()?;
    run_remote_provider_process()?;
    run_borrowed_shared_process()?;
    run_staged_lease(Arc::clone(&executor))?;
    run_borrowed_lease(Arc::clone(&executor))?;
    let unknown_token = CompletionToken {
        submission_id: SubmissionId::new(u64::MAX),
        device_epoch: provider.device_epoch(),
    };
    expect_unknown_completion(provider.wait(unknown_token, Duration::ZERO), unknown_token)?;
    println!("PASS provider_refusal slug=unknown_completion");
    println!("PASS suite provider=standalone Vulkan snapshot_parity=4");
    Ok(())
}

fn provider_error(error: ProviderError) -> Box<dyn Error> {
    format!("provider failure: {error:?}").into()
}

fn run_copy(
    provider: &VulkanComputeProvider,
    device: &Device,
    library: Library,
    encoding: &str,
    operation: u64,
) -> Result<(), Box<dyn Error>> {
    let function = library.function("copy_word")?;
    let pipeline = provider
        .compile_pipeline(
            &function,
            SemanticDigest::new("metal-smoke-fixture-v1", b"copy_word".to_vec())?,
        )
        .map_err(provider_error)?;
    let trace = make_trace(
        &pipeline,
        operation,
        Dispatch {
            kind: DispatchKind::ThreadsExact,
            grid: [1, 1, 1],
            threads_per_threadgroup: [1, 1, 1],
        },
        vec![
            (0, 8, 0x6745_2301_u32.to_le_bytes().to_vec()),
            (1, 16, 0xabab_abab_u32.to_le_bytes().to_vec()),
        ],
    )?;
    let result = submit_and_wait(provider, &trace)?;
    let expected = 0x6745_2301_u32.to_le_bytes();
    check_writeback(&trace, &result, 1, &expected)?;
    let reference = execute_copy_word(device, library)?;
    if reference.to_le_bytes() != expected {
        return Err("copy_word snapshot executor disagrees with the provider golden".into());
    }
    release_case(provider, &pipeline, &result)?;
    println!(
        "PASS provider_copy_word encoding={encoding} output={reference:#010x} writeback_offset=16 snapshot_parity=exact"
    );
    Ok(())
}

/// A submission whose observation deadline expires must be handed to the
/// shared retirement thread, not dropped in place: dropping an in-flight
/// `PendingExecution` poisons the whole Vulkan context. The second provider
/// below shares the same executor and proves the context is still usable.
fn run_timeout_reclamation(executor: Arc<VulkanExecutor>) -> Result<(), Box<dyn Error>> {
    let expiring = VulkanComputeProvider::with_executor(Arc::clone(&executor))
        .map_err(provider_error)?
        .with_async_execution(true)
        .with_observation_deadline(Duration::ZERO);
    let device = Device::new(Arc::clone(&executor) as Arc<dyn metal_api_core::ComputeExecutor>);
    let function = device
        .new_library_with_air(include_str!("../shaders/kernel_copy_word.ll"))?
        .function("copy_word")?;
    let dispatch = Dispatch {
        kind: DispatchKind::ThreadsExact,
        grid: [1, 1, 1],
        threads_per_threadgroup: [1, 1, 1],
    };
    let bindings = || {
        vec![
            (0, 8, 0x6745_2301_u32.to_le_bytes().to_vec()),
            (1, 16, 0xabab_abab_u32.to_le_bytes().to_vec()),
        ]
    };
    let digest = || SemanticDigest::new("metal-smoke-fixture-v1", b"timeout_reclamation".to_vec());
    let pipeline = expiring
        .compile_pipeline(&function, digest()?)
        .map_err(provider_error)?;
    let trace = make_trace(&pipeline, 90, dispatch, bindings())?;
    let admitted = expiring
        .capabilities()
        .validate_trace(trace.clone(), resources_for_trace(&trace)?)
        .map_err(provider_error)?;
    let submitted = expiring.submit(admitted).map_err(provider_error)?;
    let token = submitted
        .completion
        .token()
        .ok_or("timed-out submission has no token")?;
    match expiring.wait(token, Duration::ZERO) {
        Err(error)
            if error.slug == "vulkan-completion-unknown"
                && error.completion
                    == (CompletionDisposition::SubmittedUnknown { token: Some(token) }) => {}
        other => {
            return Err(
                format!("zero deadline did not publish unknown completion: {other:?}").into(),
            )
        }
    }
    expiring.release_completion(token).map_err(provider_error)?;

    let recovery =
        VulkanComputeProvider::with_executor(Arc::clone(&executor)).map_err(provider_error)?;
    let pipeline = recovery
        .compile_pipeline(&function, digest()?)
        .map_err(provider_error)?;
    let trace = make_trace(&pipeline, 91, dispatch, bindings())?;
    let result = submit_and_wait(&recovery, &trace)?;
    check_writeback(&trace, &result, 1, &0x6745_2301_u32.to_le_bytes())?;
    release_case(&recovery, &pipeline, &result)?;
    println!("PASS provider_timeout_reclamation deadline=0 context_usable=true writeback=exact");
    Ok(())
}

/// Explicit cancellation releases the observation slot without claiming device
/// retirement: `wait` keeps reporting `Cancelled`, `readback` refuses, and the
/// same provider must still accept new work once the retired submission's
/// fence signals.
fn run_cancellation(executor: Arc<VulkanExecutor>) -> Result<(), Box<dyn Error>> {
    let sink = Arc::new(RecordingSink::default());
    let provider = VulkanComputeProvider::with_executor(Arc::clone(&executor))
        .map_err(provider_error)?
        .with_async_execution(true);
    let outbox = Arc::new(CompletionOutbox::new(
        provider.device_epoch(),
        sink.clone(),
    )?);
    let provider = provider
        .with_completion_outbox(outbox)
        .map_err(provider_error)?;
    let device = Device::new(Arc::clone(&executor) as Arc<dyn metal_api_core::ComputeExecutor>);
    let function = device
        .new_library_with_air(include_str!("../shaders/kernel_copy_word.ll"))?
        .function("copy_word")?;
    let pipeline = provider
        .compile_pipeline(
            &function,
            SemanticDigest::new("metal-smoke-fixture-v1", b"cancellation".to_vec())?,
        )
        .map_err(provider_error)?;
    let dispatch = Dispatch {
        kind: DispatchKind::ThreadsExact,
        grid: [1, 1, 1],
        threads_per_threadgroup: [1, 1, 1],
    };
    let bindings = || {
        vec![
            (0, 8, 0x6745_2301_u32.to_le_bytes().to_vec()),
            (1, 16, 0xabab_abab_u32.to_le_bytes().to_vec()),
        ]
    };
    let trace = make_trace(&pipeline, 92, dispatch, bindings())?;
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources_for_trace(&trace)?)
        .map_err(provider_error)?;
    let submitted = provider.submit(admitted).map_err(provider_error)?;
    let cancelled_token = submitted
        .completion
        .token()
        .ok_or("cancelled submission has no token")?;
    match provider.cancel(cancelled_token).map_err(provider_error)? {
        CompletionDisposition::Cancelled { token: cancelled } if cancelled == cancelled_token => {}
        other => return Err(format!("cancel did not release the slot: {other:?}").into()),
    }
    match provider
        .wait(cancelled_token, Duration::ZERO)
        .map_err(provider_error)?
    {
        CompletionDisposition::Cancelled { token: waited } if waited == cancelled_token => {}
        other => return Err(format!("cancelled wait changed disposition: {other:?}").into()),
    }
    match provider.readback(cancelled_token) {
        Err(error)
            if error.slug == "completion_cancelled"
                && error.completion
                    == (CompletionDisposition::Cancelled {
                        token: cancelled_token,
                    }) => {}
        other => return Err(format!("cancelled readback was not refused: {other:?}").into()),
    }
    provider
        .release_completion(cancelled_token)
        .map_err(provider_error)?;

    // The cancelled submission was retired, not dropped in place: the provider
    // must still execute and read back new work.
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources_for_trace(&trace)?)
        .map_err(provider_error)?;
    let submitted = provider.submit(admitted).map_err(provider_error)?;
    let post_cancel_token = submitted
        .completion
        .token()
        .ok_or("post-cancel submission has no token")?;
    let observed = provider
        .wait(post_cancel_token, Duration::from_secs(10))
        .map_err(provider_error)?;
    if observed
        != (CompletionDisposition::CompletedVisible {
            token: post_cancel_token,
        })
    {
        return Err(format!("post-cancel submission did not complete: {observed:?}").into());
    }
    let readback = provider
        .readback(post_cancel_token)
        .map_err(provider_error)?;
    readback.validate_for_trace(&trace)?;
    let [writeback] = readback.writebacks.as_slice() else {
        return Err("post-cancel readback must contain exactly one writeback".into());
    };
    if writeback.bytes != 0x6745_2301_u32.to_le_bytes() {
        return Err("post-cancel readback bytes changed".into());
    }
    provider
        .release_completion(post_cancel_token)
        .map_err(provider_error)?;
    provider
        .release_pipeline(&pipeline)
        .map_err(provider_error)?;

    let observed = sink
        .messages
        .lock()
        .expect("recording completion sink")
        .iter()
        .filter_map(|message| match message {
            CompletionMessage::Token(update) => {
                Some((update.token, update.sequence.get(), update.update.clone()))
            }
            CompletionMessage::Device(_) => None,
        })
        .collect::<Vec<_>>();
    let expected = [
        (cancelled_token, 1, CompletionUpdate::Submitted),
        (cancelled_token, 2, CompletionUpdate::Cancelled),
        (post_cancel_token, 1, CompletionUpdate::Submitted),
        (post_cancel_token, 2, CompletionUpdate::CompletedVisible),
    ];
    if observed != expected {
        return Err(format!("completion outbox stream changed: {observed:?}").into());
    }
    println!(
        "PASS provider_cancellation slot_released=true context_usable=true readback=refused completion_outbox=Submitted,Cancelled,Submitted,CompletedVisible"
    );
    Ok(())
}

fn run_completion_ipc(executor: Arc<VulkanExecutor>) -> Result<(), Box<dyn Error>> {
    let (owner_transport, provider_transport) = unix::pair()?;
    let (sender, writer) = spawn_writer(provider_transport)?;
    let device = Device::new(Arc::clone(&executor) as Arc<dyn metal_api_core::ComputeExecutor>);
    let provider = VulkanComputeProvider::with_executor(executor)
        .map_err(provider_error)?
        .with_async_execution(true);
    let outbox = Arc::new(CompletionOutbox::new(
        provider.device_epoch(),
        Arc::new(sender),
    )?);
    let provider = provider
        .with_completion_outbox(outbox)
        .map_err(provider_error)?;
    let mut receiver = CompletionReceiver::new(owner_transport, provider.device_epoch())?;

    let function = device
        .new_library_with_air(include_str!("../shaders/kernel_copy_word.ll"))?
        .function("copy_word")?;
    let pipeline = provider
        .compile_pipeline(
            &function,
            SemanticDigest::new("metal-smoke-fixture-v1", b"completion_ipc".to_vec())?,
        )
        .map_err(provider_error)?;
    let trace = make_trace(
        &pipeline,
        95,
        Dispatch {
            kind: DispatchKind::ThreadsExact,
            grid: [1, 1, 1],
            threads_per_threadgroup: [1, 1, 1],
        },
        vec![
            (0, 8, 0x6745_2301_u32.to_le_bytes().to_vec()),
            (1, 16, 0xabab_abab_u32.to_le_bytes().to_vec()),
        ],
    )?;
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources_for_trace(&trace)?)
        .map_err(provider_error)?;
    let submitted = provider.submit(admitted).map_err(provider_error)?;
    let token = submitted
        .completion
        .token()
        .ok_or("ipc submission has no token")?;
    if receiver.recv()? != MirrorOutcome::Applied {
        return Err("ipc receiver did not apply admission".into());
    }
    let observed = provider
        .wait(token, Duration::from_secs(10))
        .map_err(provider_error)?;
    if observed != (CompletionDisposition::CompletedVisible { token }) {
        return Err(format!("ipc submission did not complete: {observed:?}").into());
    }
    if receiver.recv()? != MirrorOutcome::Applied {
        return Err("ipc receiver did not apply completion".into());
    }

    let lease_id = LeaseId::new(95);
    let mut ledger = LeaseLedger::new();
    ledger.register(LeaseReservation {
        lease: BufferLease {
            lease_id,
            allocation_id: AllocationId::new(95),
            owner_epoch: provider.device_epoch(),
        },
        offset: 0,
        length: 64,
    })?;
    ledger.bind(lease_id, token)?;
    if receiver.observe_into(&mut ledger, token)? != LeaseObservation::Retired {
        return Err("ipc completion did not retire the lease".into());
    }
    if !ledger.release_ready(lease_id) {
        return Err("ipc lease was not release-ready".into());
    }

    provider.release_completion(token).map_err(provider_error)?;
    provider
        .release_pipeline(&pipeline)
        .map_err(provider_error)?;
    drop(provider);
    drop(receiver);
    writer.join().map_err(|_| "ipc writer panicked")??;
    println!(
        "PASS provider_completion_ipc transport=unix outbox=Submitted,CompletedVisible lease=retired"
    );
    Ok(())
}

/// Provider half of the two-process completion test.
///
/// The child owns the Vulkan device, connects to the owner's listener and
/// publishes admission and the terminal transition through the real outbox and
/// writer thread. The handshake line tells the owner which device epoch and
/// submission identity to mirror before any frame is applied.
pub fn run_completion_child(socket: &std::ffi::OsStr) -> Result<(), Box<dyn Error>> {
    let executor = VulkanExecutor::new()?;
    let device = Device::new(Arc::clone(&executor) as Arc<dyn metal_api_core::ComputeExecutor>);
    let provider = VulkanComputeProvider::with_executor(executor)
        .map_err(provider_error)?
        .with_async_execution(true);
    let transport = unix::connect(socket)?;
    let (sender, writer) = spawn_writer(transport)?;
    let outbox = Arc::new(CompletionOutbox::new(
        provider.device_epoch(),
        Arc::new(sender),
    )?);
    let provider = provider
        .with_completion_outbox(outbox)
        .map_err(provider_error)?;

    let function = device
        .new_library_with_air(include_str!("../shaders/kernel_copy_word.ll"))?
        .function("copy_word")?;
    let pipeline = provider
        .compile_pipeline(
            &function,
            SemanticDigest::new("metal-smoke-fixture-v1", b"completion_ipc_process".to_vec())?,
        )
        .map_err(provider_error)?;
    let trace = make_trace(
        &pipeline,
        96,
        Dispatch {
            kind: DispatchKind::ThreadsExact,
            grid: [1, 1, 1],
            threads_per_threadgroup: [1, 1, 1],
        },
        vec![
            (0, 8, 0x6745_2301_u32.to_le_bytes().to_vec()),
            (1, 16, 0xabab_abab_u32.to_le_bytes().to_vec()),
        ],
    )?;
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources_for_trace(&trace)?)
        .map_err(provider_error)?;
    let submitted = provider.submit(admitted).map_err(provider_error)?;
    let token = submitted
        .completion
        .token()
        .ok_or("completion child submission has no token")?;
    println!(
        "handshake epoch={} submission={}",
        token.device_epoch.get(),
        token.submission_id.get()
    );
    std::io::stdout().flush()?;

    let observed = provider
        .wait(token, Duration::from_secs(10))
        .map_err(provider_error)?;
    if observed != (CompletionDisposition::CompletedVisible { token }) {
        return Err(format!("completion child submission did not complete: {observed:?}").into());
    }
    provider.release_completion(token).map_err(provider_error)?;
    provider
        .release_pipeline(&pipeline)
        .map_err(provider_error)?;
    drop(provider);
    writer
        .join()
        .map_err(|_| "completion child writer panicked")??;
    println!(
        "PASS completion_child epoch={} submission={} completed=true",
        token.device_epoch.get(),
        token.submission_id.get()
    );
    Ok(())
}

/// Provider half of the owner-command test.
///
/// The child owns the Vulkan device, serves owner commands on the command
/// socket and publishes admission and terminal notifications on the
/// completion socket. It never submits work on its own: every compile,
/// submit, wait and readback below is initiated by the owner process.
pub fn run_provider_command_child(
    command_socket: &std::ffi::OsStr,
    completion_socket: &std::ffi::OsStr,
) -> Result<(), Box<dyn Error>> {
    let executor = VulkanExecutor::new()?;
    let provider = VulkanComputeProvider::with_executor(executor)
        .map_err(provider_error)?
        .with_async_execution(true);
    let epoch = provider.device_epoch();
    let completion_transport = unix::connect(completion_socket)?;
    let (sender, writer) = spawn_writer(completion_transport)?;
    let outbox = Arc::new(CompletionOutbox::new(epoch, Arc::new(sender))?);
    let provider = provider
        .with_completion_outbox(outbox)
        .map_err(provider_error)?;
    let mut transport = command_unix::connect(command_socket)?;
    serve_provider_unix(&provider, &mut transport)?;
    drop(provider);
    writer
        .join()
        .map_err(|_| "provider command writer panicked")??;
    println!("PASS command_child epoch={} served=true", epoch.get());
    Ok(())
}

/// Owner half of the two-process completion test.
///
/// The parent process owns no provider in this case: it listens on a Unix
/// socket, spawns the provider child and retires a lease from the mirrored
/// completion stream alone.
fn run_completion_ipc_process() -> Result<(), Box<dyn Error>> {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "metal-smoke-completion-{}-{unique}.sock",
        std::process::id()
    ));
    let listener = unix::UnixListenerTransport::bind(&path)?;
    let mut child = Command::new(std::env::current_exe()?)
        .arg("--completion-child")
        .arg(&path)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or("completion child stdout was not piped")?;
    let mut lines = BufReader::new(stdout).lines();
    let handshake = lines
        .next()
        .ok_or("completion child exited before its handshake")??;
    let (device_epoch, token) = parse_handshake(&handshake)?;

    let transport = listener.accept()?;
    transport.set_read_timeout(Some(Duration::from_secs(30)))?;
    let mut receiver = CompletionReceiver::new(transport, device_epoch)?;
    if receiver.recv()? != MirrorOutcome::Applied {
        return Err("completion process receiver did not apply admission".into());
    }
    if receiver.recv()? != MirrorOutcome::Applied {
        return Err("completion process receiver did not apply completion".into());
    }
    if receiver.applied() != 2 || receiver.ignored() != 0 {
        return Err(format!(
            "completion process mirror counted applied={} ignored={}",
            receiver.applied(),
            receiver.ignored()
        )
        .into());
    }

    let lease_id = LeaseId::new(96);
    let mut ledger = LeaseLedger::new();
    ledger.register(LeaseReservation {
        lease: BufferLease {
            lease_id,
            allocation_id: AllocationId::new(96),
            owner_epoch: device_epoch,
        },
        offset: 0,
        length: 64,
    })?;
    ledger.bind(lease_id, token)?;
    if receiver.observe_into(&mut ledger, token)? != LeaseObservation::Retired {
        return Err("completion process did not retire the lease".into());
    }
    if !ledger.release_ready(lease_id) {
        return Err("completion process lease was not release-ready".into());
    }

    let status = child.wait()?;
    for line in lines {
        println!("child: {}", line?);
    }
    let _ = std::fs::remove_file(&path);
    if !status.success() {
        return Err(format!("completion child exited with {status}").into());
    }
    println!(
        "PASS provider_completion_ipc_process owner=parent provider=child transport=unix outbox=Submitted,CompletedVisible lease=retired"
    );
    Ok(())
}

/// Owner half of the owner-command test.
///
/// The parent owns no provider. It compiles a reviewed shader on the provider
/// child through the command channel, builds and submits the trace itself,
/// mirrors the provider's completion stream on a second connection, and reads
/// the result back over the command channel. The child never submits on its
/// own, so this proves the owner can remotely drive a provider.
fn run_remote_provider_process() -> Result<(), Box<dyn Error>> {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let command_path = std::env::temp_dir().join(format!(
        "metal-smoke-command-{}-{unique}.sock",
        std::process::id()
    ));
    let completion_path = std::env::temp_dir().join(format!(
        "metal-smoke-command-completion-{}-{unique}.sock",
        std::process::id()
    ));
    let command_listener = command_unix::UnixListenerCommandTransport::bind(&command_path)?;
    let completion_listener = unix::UnixListenerTransport::bind(&completion_path)?;
    let mut child = Command::new(std::env::current_exe()?)
        .arg("--command-child")
        .arg(&command_path)
        .arg(&completion_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or("provider command child stdout was not piped")?;
    let mut lines = BufReader::new(stdout).lines();

    let command_transport = command_listener.accept()?;
    command_transport.set_read_timeout(Some(Duration::from_secs(30)))?;
    let completion_transport = completion_listener.accept()?;
    completion_transport.set_read_timeout(Some(Duration::from_secs(30)))?;

    let remote = RemoteProvider::connect(command_transport)?;
    let epoch = remote.device_epoch();
    let mut receiver = CompletionReceiver::new(completion_transport, epoch)?;
    let compile = PipelineCompileRequest {
        entry_name: "copy_word".into(),
        logical_digest: SemanticDigest::new(
            "metal-smoke-fixture-v1",
            b"remote_provider_command".to_vec(),
        )?,
        source: ShaderSource::SanitizedLl(
            include_str!("../shaders/kernel_copy_word.ll").to_string(),
        ),
    };
    let pipeline = remote.compile(compile).map_err(provider_error)?;
    if remote.health() != ProviderHealth::Usable {
        return Err(format!(
            "remote provider health is not usable: {:?}",
            remote.health()
        )
        .into());
    }
    let lease_id = LeaseId::new(96);
    let reservation = LeaseReservation {
        lease: BufferLease {
            lease_id,
            allocation_id: AllocationId::new(100),
            owner_epoch: epoch,
        },
        offset: 8,
        length: 4,
    };
    let word = 0x6745_2301_u32.to_le_bytes().to_vec();
    remote
        .import_staged_lease(StagedLease::new(reservation, word.clone())?)
        .map_err(provider_error)?;
    let mut trace = make_trace(
        &pipeline,
        501,
        Dispatch {
            kind: DispatchKind::ThreadsExact,
            grid: [1, 1, 1],
            threads_per_threadgroup: [1, 1, 1],
        },
        vec![(0, 8, word.clone()), (1, 16, vec![0; 4])],
    )?;
    trace.passes[0].buffers[0].source = BufferSource::StagedLease(lease_id);
    let mut resources = resources_for_trace(&trace)?;
    resources.insert_lease(reservation)?;
    let admitted = remote
        .capabilities()
        .validate_trace(trace.clone(), resources.clone())
        .map_err(provider_error)?;
    let submitted = remote.submit(admitted).map_err(provider_error)?;
    let token = submitted
        .completion
        .token()
        .ok_or("remote command submission has no token")?;
    if !matches!(
        submitted.completion,
        CompletionDisposition::Submitted { .. }
    ) {
        return Err(format!(
            "remote provider did not acknowledge submission: {:?}",
            submitted.completion
        )
        .into());
    }
    if receiver.recv()? != MirrorOutcome::Applied {
        return Err("remote command receiver did not apply admission".into());
    }
    let observed = remote
        .wait(token, Duration::from_secs(10))
        .map_err(provider_error)?;
    if observed != (CompletionDisposition::CompletedVisible { token }) {
        return Err(format!("remote provider did not complete: {observed:?}").into());
    }
    if receiver.recv()? != MirrorOutcome::Applied {
        return Err("remote command receiver did not apply completion".into());
    }
    let readback = remote.readback(token).map_err(provider_error)?;
    let result = ProviderSubmission {
        completion: readback.completion,
        writebacks: readback.writebacks,
    };
    result.validate_for_trace(&trace)?;
    check_writeback(&trace, &result, 1, &word)?;
    let mut ledger = LeaseLedger::new();
    ledger.register(reservation)?;
    ledger.bind(lease_id, token)?;
    if receiver.observe_into(&mut ledger, token)? != LeaseObservation::Retired {
        return Err("remote command completion did not retire the staged lease".into());
    }
    if !ledger.release_ready(lease_id) {
        return Err("remote command staged lease was not release-ready".into());
    }
    remote.release_completion(token).map_err(provider_error)?;
    remote
        .release_staged_lease(lease_id)
        .map_err(provider_error)?;
    let admitted = remote
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .map_err(provider_error)?;
    let refused = remote.submit(admitted).unwrap_err();
    if refused.slug != "lease_not_imported" {
        return Err(
            format!("remote provider did not refuse the released lease: {refused:?}").into(),
        );
    }

    // Descriptor-backed no-copy lease over the same command connection.
    let mut mapping = shared::SharedMemory::create(BORROWED_SHARED_SIZE)?;
    mapping.as_mut_slice().fill(0xcd);
    let borrowed_lease_id = LeaseId::new(95);
    let borrowed_reservation = LeaseReservation {
        lease: BufferLease {
            lease_id: borrowed_lease_id,
            allocation_id: AllocationId::new(BORROWED_SHARED_ALLOCATION_ID),
            owner_epoch: epoch,
        },
        offset: 0,
        length: BORROWED_SHARED_LENGTH,
    };
    remote
        .import_borrowed_lease(borrowed_reservation, &mapping)
        .map_err(provider_error)?;
    let borrowed_trace = borrowed_lease_trace(
        epoch,
        &pipeline,
        borrowed_lease_id,
        BufferAccess::Write,
        BufferSource::OwnedBytes(BORROWED_SHARED_GPU_WORD.to_le_bytes().to_vec()),
        601,
        602,
    );
    let borrowed_resources = borrowed_lease_resources(epoch, borrowed_reservation)?;
    let admitted = remote
        .capabilities()
        .validate_trace(borrowed_trace.clone(), borrowed_resources.clone())
        .map_err(provider_error)?;
    let submitted = remote.submit(admitted).map_err(provider_error)?;
    let borrowed_token = submitted
        .completion
        .token()
        .ok_or("remote borrowed submission has no token")?;
    if receiver.recv()? != MirrorOutcome::Applied {
        return Err("remote command receiver did not apply borrowed admission".into());
    }
    let observed = remote
        .wait(borrowed_token, Duration::from_secs(10))
        .map_err(provider_error)?;
    if observed
        != (CompletionDisposition::CompletedVisible {
            token: borrowed_token,
        })
    {
        return Err(format!("remote borrowed lease did not complete: {observed:?}").into());
    }
    if receiver.recv()? != MirrorOutcome::Applied {
        return Err("remote command receiver did not apply borrowed completion".into());
    }
    let readback = remote.readback(borrowed_token).map_err(provider_error)?;
    let borrowed_result = ProviderSubmission {
        completion: readback.completion,
        writebacks: readback.writebacks,
    };
    borrowed_result.validate_for_trace(&borrowed_trace)?;
    check_writeback(
        &borrowed_trace,
        &borrowed_result,
        1,
        &BORROWED_SHARED_GPU_WORD.to_le_bytes(),
    )?;
    if mapping.as_slice()[..4] != BORROWED_SHARED_GPU_WORD.to_le_bytes() {
        return Err(format!(
            "remote borrowed mapping did not observe the GPU write: {:02x?}",
            &mapping.as_slice()[..4]
        )
        .into());
    }
    if mapping.as_slice()[4..].iter().any(|byte| *byte != 0xcd) {
        return Err("remote borrowed mapping guards changed".into());
    }
    let mut ledger = LeaseLedger::new();
    ledger.register(borrowed_reservation)?;
    ledger.bind(borrowed_lease_id, borrowed_token)?;
    if receiver.observe_into(&mut ledger, borrowed_token)? != LeaseObservation::Retired {
        return Err("remote command completion did not retire the borrowed lease".into());
    }
    if !ledger.release_ready(borrowed_lease_id) {
        return Err("remote command borrowed lease was not release-ready".into());
    }
    remote
        .release_completion(borrowed_token)
        .map_err(provider_error)?;
    remote
        .release_borrowed_lease(borrowed_lease_id)
        .map_err(provider_error)?;
    let admitted = remote
        .capabilities()
        .validate_trace(borrowed_trace.clone(), borrowed_resources)
        .map_err(provider_error)?;
    let refused = remote.submit(admitted).unwrap_err();
    if refused.slug != "lease_not_imported" {
        return Err(format!(
            "remote provider did not refuse the released borrowed lease: {refused:?}"
        )
        .into());
    }
    if receiver.applied() != 4 || receiver.ignored() != 0 {
        return Err(format!(
            "remote command mirror counted applied={} ignored={}",
            receiver.applied(),
            receiver.ignored()
        )
        .into());
    }
    remote.release_pipeline(&pipeline).map_err(provider_error)?;
    drop(remote);

    let status = child.wait()?;
    for line in &mut lines {
        println!("child: {}", line?);
    }
    let _ = std::fs::remove_file(&command_path);
    let _ = std::fs::remove_file(&completion_path);
    if !status.success() {
        return Err(format!("provider command child exited with {status}").into());
    }
    println!(
        "PASS provider_command_process owner=parent provider=child transport=unix commands=health,compile,import_lease,import_borrowed,submit,wait,readback,release completion=mirrored writeback=exact lease=retired,refused borrowed=retired,in_place"
    );
    Ok(())
}

/// Provider half of the two-process borrowed no-copy test.
///
/// The child receives an owner mapping over `SCM_RIGHTS`, imports the same
/// physical pages as a borrowed lease and submits one read and one write
/// through them. The owner proves the write landed in its own mapping without
/// any writeback copy.
pub fn run_borrowed_shared_child(socket: &std::ffi::OsStr) -> Result<(), Box<dyn Error>> {
    let stream = UnixStream::connect(socket)?;
    let descriptor = shared::recv_fd(&stream)?;
    let mapping = shared::SharedMemory::from_owned_fd(descriptor)?;
    let transport = unix::from_stream(stream)?;

    let executor = VulkanExecutor::new()?;
    let device = Device::new(Arc::clone(&executor) as Arc<dyn metal_api_core::ComputeExecutor>);
    let provider = VulkanComputeProvider::with_executor(executor).map_err(provider_error)?;
    let alignment = provider.no_copy_alignment();
    if alignment == 0 {
        return Err(
            "borrowed shared child: provider does not advertise VK_EXT_external_memory_host".into(),
        );
    }
    if !provider
        .capabilities()
        .storage_modes
        .contains(&StorageMode::BorrowedNoCopy)
    {
        return Err("borrowed shared child: provider does not advertise BorrowedNoCopy".into());
    }
    if !(mapping.as_ptr() as usize).is_multiple_of(alignment as usize) {
        return Err(format!(
            "borrowed shared child: mapping {:p} is not aligned to {alignment}",
            mapping.as_ptr()
        )
        .into());
    }
    let lease_id = LeaseId::new(BORROWED_SHARED_LEASE_ID);
    let reservation = LeaseReservation {
        lease: BufferLease {
            lease_id,
            allocation_id: AllocationId::new(BORROWED_SHARED_ALLOCATION_ID),
            owner_epoch: provider.device_epoch(),
        },
        offset: 0,
        length: BORROWED_SHARED_LENGTH,
    };
    // SAFETY: `mapping` outlives the import and stays mapped until the lease
    // is released below.
    unsafe {
        provider
            .import_borrowed_lease(BorrowedLease::new(reservation, mapping.as_ptr() as usize)?)
            .map_err(provider_error)?;
    }

    let (sender, writer) = spawn_writer(transport)?;
    let outbox = Arc::new(CompletionOutbox::new(
        provider.device_epoch(),
        Arc::new(sender),
    )?);
    let provider = provider
        .with_completion_outbox(outbox)
        .map_err(provider_error)?;

    let function = device
        .new_library_with_air(include_str!("../shaders/kernel_copy_word.ll"))?
        .function("copy_word")?;
    let pipeline = provider
        .compile_pipeline(
            &function,
            SemanticDigest::new("metal-smoke-fixture-v1", b"borrowed_shared".to_vec())?,
        )
        .map_err(provider_error)?;

    let read_trace = borrowed_lease_trace(
        provider.device_epoch(),
        &pipeline,
        lease_id,
        BufferAccess::Read,
        BufferSource::OwnedBytes(vec![0; 4]),
        497,
        498,
    );
    let read_resources = borrowed_lease_resources(provider.device_epoch(), reservation)?;
    let admitted = provider
        .capabilities()
        .validate_trace(read_trace.clone(), read_resources)
        .map_err(provider_error)?;
    let read_result = provider.submit(admitted).map_err(provider_error)?;
    read_result.validate_for_trace(&read_trace)?;
    let read_token = read_result
        .completion
        .token()
        .ok_or("borrowed shared read has no token")?;

    let write_trace = borrowed_lease_trace(
        provider.device_epoch(),
        &pipeline,
        lease_id,
        BufferAccess::Write,
        BufferSource::OwnedBytes(BORROWED_SHARED_GPU_WORD.to_le_bytes().to_vec()),
        499,
        500,
    );
    let write_resources = borrowed_lease_resources(provider.device_epoch(), reservation)?;
    let admitted = provider
        .capabilities()
        .validate_trace(write_trace.clone(), write_resources)
        .map_err(provider_error)?;
    let write_result = provider.submit(admitted).map_err(provider_error)?;
    write_result.validate_for_trace(&write_trace)?;
    let write_token = write_result
        .completion
        .token()
        .ok_or("borrowed shared write has no token")?;

    for token in [read_token, write_token] {
        println!(
            "handshake epoch={} submission={}",
            token.device_epoch.get(),
            token.submission_id.get()
        );
    }
    std::io::stdout().flush()?;

    let observed = provider
        .wait(read_token, Duration::from_secs(10))
        .map_err(provider_error)?;
    if observed != (CompletionDisposition::CompletedVisible { token: read_token }) {
        return Err(format!("borrowed shared read did not complete: {observed:?}").into());
    }
    check_writeback(
        &read_trace,
        &read_result,
        1,
        &BORROWED_SHARED_OWNER_WORD.to_le_bytes(),
    )?;

    let observed = provider
        .wait(write_token, Duration::from_secs(10))
        .map_err(provider_error)?;
    if observed != (CompletionDisposition::CompletedVisible { token: write_token }) {
        return Err(format!("borrowed shared write did not complete: {observed:?}").into());
    }
    check_writeback(
        &write_trace,
        &write_result,
        1,
        &BORROWED_SHARED_GPU_WORD.to_le_bytes(),
    )?;
    if mapping.as_slice()[..4] != BORROWED_SHARED_GPU_WORD.to_le_bytes() {
        return Err(format!(
            "borrowed shared child mapping did not observe the GPU write: {:02x?}",
            &mapping.as_slice()[..4]
        )
        .into());
    }

    let mut ledger = LeaseLedger::new();
    ledger.register(reservation)?;
    for token in [read_token, write_token] {
        ledger.bind(lease_id, token)?;
        if ledger.observe(token, CompletionDisposition::CompletedVisible { token })?
            != LeaseObservation::Retired
        {
            return Err("borrowed shared completion did not retire the lease".into());
        }
    }
    if !ledger.release_ready(lease_id) {
        return Err("borrowed shared lease was not release-ready".into());
    }
    provider
        .release_completion(read_token)
        .map_err(provider_error)?;
    provider
        .release_completion(write_token)
        .map_err(provider_error)?;
    provider
        .release_borrowed_lease(lease_id)
        .map_err(provider_error)?;
    provider
        .release_pipeline(&pipeline)
        .map_err(provider_error)?;
    drop(provider);
    writer
        .join()
        .map_err(|_| "borrowed shared child writer panicked")??;
    println!(
        "PASS borrowed_shared_child lease={BORROWED_SHARED_LEASE_ID} copy_in=owner_visible copy_out=in_place retired=true"
    );
    Ok(())
}

/// Owner half of the two-process borrowed no-copy test.
///
/// The parent keeps no Vulkan provider. It creates the mapping, passes the
/// descriptor with `SCM_RIGHTS`, mirrors the completion stream and then reads
/// its own pages to prove the child's GPU write happened in place.
fn run_borrowed_shared_process() -> Result<(), Box<dyn Error>> {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "metal-smoke-borrowed-{}-{unique}.sock",
        std::process::id()
    ));
    let listener = unix::UnixListenerTransport::bind(&path)?;
    let mut mapping = shared::SharedMemory::create(BORROWED_SHARED_SIZE)?;
    mapping.as_mut_slice().fill(0xcd);
    mapping.as_mut_slice()[..4].copy_from_slice(&BORROWED_SHARED_OWNER_WORD.to_le_bytes());

    let mut child = Command::new(std::env::current_exe()?)
        .arg("--borrowed-shared-child")
        .arg(&path)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or("borrowed shared child stdout was not piped")?;
    let mut lines = BufReader::new(stdout).lines();

    let transport = listener.accept()?;
    transport.set_read_timeout(Some(Duration::from_secs(30)))?;
    shared::send_fd(transport.writer(), mapping.descriptor())?;

    let mut device_epoch = None;
    let mut tokens = Vec::new();
    for _ in 0..2 {
        let line = lines
            .next()
            .ok_or("borrowed shared child exited before its handshake")??;
        let (epoch, token) = parse_handshake(&line)?;
        match device_epoch {
            None => device_epoch = Some(epoch),
            Some(expected) if expected != epoch => {
                return Err("borrowed shared child changed its device epoch".into())
            }
            Some(_) => {}
        }
        tokens.push(token);
    }
    let device_epoch = device_epoch.ok_or("borrowed shared child produced no handshake")?;

    let mut receiver = CompletionReceiver::new(transport, device_epoch)?;
    for _ in 0..4 {
        if receiver.recv()? != MirrorOutcome::Applied {
            return Err("borrowed shared receiver did not apply a completion frame".into());
        }
    }
    if receiver.applied() != 4 || receiver.ignored() != 0 {
        return Err(format!(
            "borrowed shared mirror counted applied={} ignored={}",
            receiver.applied(),
            receiver.ignored()
        )
        .into());
    }

    let lease_id = LeaseId::new(BORROWED_SHARED_LEASE_ID);
    let mut ledger = LeaseLedger::new();
    ledger.register(LeaseReservation {
        lease: BufferLease {
            lease_id,
            allocation_id: AllocationId::new(BORROWED_SHARED_ALLOCATION_ID),
            owner_epoch: device_epoch,
        },
        offset: 0,
        length: BORROWED_SHARED_LENGTH,
    })?;
    for token in &tokens {
        ledger.bind(lease_id, *token)?;
        if receiver.observe_into(&mut ledger, *token)? != LeaseObservation::Retired {
            return Err("borrowed shared completion did not retire the lease".into());
        }
    }
    if !ledger.release_ready(lease_id) {
        return Err("borrowed shared lease was not release-ready".into());
    }

    let status = child.wait()?;
    for line in lines {
        println!("child: {}", line?);
    }
    let _ = std::fs::remove_file(&path);
    if !status.success() {
        return Err(format!("borrowed shared child exited with {status}").into());
    }
    if mapping.as_slice()[..4] != BORROWED_SHARED_GPU_WORD.to_le_bytes() {
        return Err(format!(
            "owner mapping did not observe the child GPU write: {:02x?}",
            &mapping.as_slice()[..4]
        )
        .into());
    }
    if mapping.as_slice()[4..].iter().any(|byte| *byte != 0xcd) {
        return Err("owner mapping guards changed".into());
    }
    println!(
        "PASS provider_borrowed_shared_process owner=parent provider=child transport=scm_rights copy_in=owner_visible copy_out=in_place retired=true"
    );
    Ok(())
}

fn parse_handshake(line: &str) -> Result<(DeviceEpoch, CompletionToken), Box<dyn Error>> {
    let mut epoch = None;
    let mut submission = None;
    for field in line.split_whitespace() {
        if let Some(value) = field.strip_prefix("epoch=") {
            epoch = Some(value.parse::<u64>()?);
        } else if let Some(value) = field.strip_prefix("submission=") {
            submission = Some(value.parse::<u64>()?);
        }
    }
    let device_epoch = DeviceEpoch::new(epoch.ok_or("completion handshake is missing epoch")?);
    let submission_id =
        SubmissionId::new(submission.ok_or("completion handshake is missing submission")?);
    Ok((
        device_epoch,
        CompletionToken {
            device_epoch,
            submission_id,
        },
    ))
}

/// Import owner-issued bytes for one lease, execute a view backed by that
/// lease, retire it through the owner ledger and refuse it after release.
fn run_staged_lease(executor: Arc<VulkanExecutor>) -> Result<(), Box<dyn Error>> {
    let device = Device::new(Arc::clone(&executor) as Arc<dyn metal_api_core::ComputeExecutor>);
    let provider = VulkanComputeProvider::with_executor(executor).map_err(provider_error)?;
    let function = device
        .new_library_with_air(include_str!("../shaders/kernel_copy_word.ll"))?
        .function("copy_word")?;
    let pipeline = provider
        .compile_pipeline(
            &function,
            SemanticDigest::new("metal-smoke-fixture-v1", b"staged_lease".to_vec())?,
        )
        .map_err(provider_error)?;

    let lease_id = LeaseId::new(97);
    let reservation = LeaseReservation {
        lease: BufferLease {
            lease_id,
            allocation_id: AllocationId::new(197),
            owner_epoch: provider.device_epoch(),
        },
        offset: 0,
        length: 8,
    };
    let mut input = 0x6745_2301_u32.to_le_bytes().to_vec();
    input.extend_from_slice(&[0_u8; 4]);
    provider
        .import_staged_lease(StagedLease::new(reservation, input)?)
        .map_err(provider_error)?;

    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(97),
        pipelines: vec![pipeline.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![ComputePass {
            pipeline: pipeline.pipeline_id,
            buffers: vec![
                BufferView {
                    view_id: ViewId::new(297),
                    metal_binding: 0,
                    allocation_id: AllocationId::new(197),
                    offset: 0,
                    length: 4,
                    access: BufferAccess::Read,
                    attribute_stride: None,
                    source: BufferSource::StagedLease(lease_id),
                },
                BufferView {
                    view_id: ViewId::new(298),
                    metal_binding: 1,
                    allocation_id: AllocationId::new(198),
                    offset: 0,
                    length: 4,
                    access: BufferAccess::Write,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(vec![0xab; 4]),
                },
            ],
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            },
        }],
        completion_policy: CompletionPolicy::HostReadback,
    };
    let mut resources = ResourceTableSnapshot::new();
    resources.insert_allocation(AllocationRecord {
        allocation_id: AllocationId::new(197),
        owner_epoch: provider.device_epoch(),
        size: 16,
    })?;
    resources.insert_allocation(AllocationRecord {
        allocation_id: AllocationId::new(198),
        owner_epoch: provider.device_epoch(),
        size: 32,
    })?;
    resources.insert_lease(reservation)?;

    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources.clone())
        .map_err(provider_error)?;
    let result = provider.submit(admitted).map_err(provider_error)?;
    result.validate_for_trace(&trace)?;
    let CompletionDisposition::CompletedVisible { token } = result.completion else {
        return Err(format!(
            "staged lease submission did not complete: {:?}",
            result.completion
        )
        .into());
    };
    check_writeback(&trace, &result, 1, &0x6745_2301_u32.to_le_bytes())?;

    let mut ledger = LeaseLedger::new();
    ledger.register(reservation)?;
    ledger.bind(lease_id, token)?;
    if ledger.observe(token, result.completion)? != LeaseObservation::Retired {
        return Err("staged lease completion did not retire the lease".into());
    }
    if !ledger.release_ready(lease_id) {
        return Err("staged lease was not release-ready".into());
    }

    provider
        .release_staged_lease(lease_id)
        .map_err(provider_error)?;
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .map_err(provider_error)?;
    let error = provider
        .submit(admitted)
        .expect_err("released staged lease must be refused");
    if error.slug != "lease_not_imported" {
        return Err(format!("released staged lease refused with {}", error.slug).into());
    }

    provider.release_completion(token).map_err(provider_error)?;
    provider
        .release_pipeline(&pipeline)
        .map_err(provider_error)?;
    println!(
        "PASS provider_staged_lease lease=97 writeback=exact retired=true refusal=lease_not_imported"
    );
    Ok(())
}

/// Owner host memory imported without copying. Two submissions prove the
/// provider reads and writes the live mapping: one mutates owner memory after
/// import and observes the new value on the GPU, and one writes through the
/// GPU directly into owner memory.
fn run_borrowed_lease(executor: Arc<VulkanExecutor>) -> Result<(), Box<dyn Error>> {
    let device = Device::new(Arc::clone(&executor) as Arc<dyn metal_api_core::ComputeExecutor>);
    let provider = VulkanComputeProvider::with_executor(executor).map_err(provider_error)?;
    let alignment = provider.no_copy_alignment();
    if alignment == 0 {
        return Err("provider does not advertise VK_EXT_external_memory_host".into());
    }
    if !provider
        .capabilities()
        .storage_modes
        .contains(&StorageMode::BorrowedNoCopy)
    {
        return Err("provider does not advertise the borrowed no-copy storage mode".into());
    }
    let function = device
        .new_library_with_air(include_str!("../shaders/kernel_copy_word.ll"))?
        .function("copy_word")?;
    let pipeline = provider
        .compile_pipeline(
            &function,
            SemanticDigest::new("metal-smoke-fixture-v1", b"borrowed_lease".to_vec())?,
        )
        .map_err(provider_error)?;

    let lease_id = LeaseId::new(98);
    let reservation = LeaseReservation {
        lease: BufferLease {
            lease_id,
            allocation_id: AllocationId::new(298),
            owner_epoch: provider.device_epoch(),
        },
        offset: 0,
        length: 64,
    };
    let mut owner = AlignedBuffer::new(64, alignment as usize)?;
    owner.as_mut_slice().fill(0xcd);
    owner.as_mut_slice()[..4].copy_from_slice(&0xaaaa_aaaa_u32.to_le_bytes());
    // A misaligned owner pointer must be refused before any Vulkan import.
    let misaligned = unsafe {
        provider.import_borrowed_lease(BorrowedLease::new(
            reservation,
            owner.as_ptr() as usize + 1,
        )?)
    }
    .expect_err("misaligned borrowed lease must be refused");
    if misaligned.slug != "lease_alignment_unsupported" {
        return Err(format!("misaligned borrowed lease refused with {}", misaligned.slug).into());
    }
    // SAFETY: `owner` stays alive until both submissions retire and the
    // provider releases the import below.
    unsafe {
        provider
            .import_borrowed_lease(BorrowedLease::new(reservation, owner.as_ptr() as usize)?)
            .map_err(provider_error)?;
    }

    // The owner may change its mapping after import. A provider that had
    // snapshotted the bytes would observe the old word.
    owner.as_mut_slice()[..4].copy_from_slice(&0xbbbb_bbbb_u32.to_le_bytes());
    let read_trace = borrowed_lease_trace(
        provider.device_epoch(),
        &pipeline,
        lease_id,
        BufferAccess::Read,
        BufferSource::OwnedBytes(vec![0; 4]),
        397,
        398,
    );
    let read_resources = borrowed_lease_resources(provider.device_epoch(), reservation)?;
    let admitted = provider
        .capabilities()
        .validate_trace(read_trace.clone(), read_resources.clone())
        .map_err(provider_error)?;
    let read_result = provider.submit(admitted).map_err(provider_error)?;
    read_result.validate_for_trace(&read_trace)?;
    let CompletionDisposition::CompletedVisible { token: read_token } = read_result.completion
    else {
        return Err(format!(
            "borrowed lease read submission did not complete: {:?}",
            read_result.completion
        )
        .into());
    };
    check_writeback(&read_trace, &read_result, 1, &0xbbbb_bbbb_u32.to_le_bytes())?;

    let word = 0x1234_5678_u32;
    let write_trace = borrowed_lease_trace(
        provider.device_epoch(),
        &pipeline,
        lease_id,
        BufferAccess::Write,
        BufferSource::OwnedBytes(word.to_le_bytes().to_vec()),
        399,
        400,
    );
    let write_resources = borrowed_lease_resources(provider.device_epoch(), reservation)?;
    let admitted = provider
        .capabilities()
        .validate_trace(write_trace.clone(), write_resources.clone())
        .map_err(provider_error)?;
    let write_result = provider.submit(admitted).map_err(provider_error)?;
    write_result.validate_for_trace(&write_trace)?;
    let CompletionDisposition::CompletedVisible { token: write_token } = write_result.completion
    else {
        return Err(format!(
            "borrowed lease write submission did not complete: {:?}",
            write_result.completion
        )
        .into());
    };
    // The GPU wrote through the import; no writeback was applied to `owner`.
    if owner.as_slice()[..4] != word.to_le_bytes() {
        return Err(format!(
            "borrowed lease did not write in place: {:02x?}",
            &owner.as_slice()[..4]
        )
        .into());
    }
    check_writeback(&write_trace, &write_result, 1, &word.to_le_bytes())?;

    let mut ledger = LeaseLedger::new();
    ledger.register(reservation)?;
    for token in [read_token, write_token] {
        ledger.bind(lease_id, token)?;
        if ledger.observe(token, CompletionDisposition::CompletedVisible { token })?
            != LeaseObservation::Retired
        {
            return Err("borrowed lease completion did not retire the lease".into());
        }
    }
    if !ledger.release_ready(lease_id) {
        return Err("borrowed lease was not release-ready".into());
    }
    if provider.borrowed_registry().outstanding(lease_id) != Some(0) {
        return Err("borrowed lease retains were not retired".into());
    }
    provider
        .release_borrowed_lease(lease_id)
        .map_err(provider_error)?;

    let admitted = provider
        .capabilities()
        .validate_trace(write_trace.clone(), write_resources)
        .map_err(provider_error)?;
    let error = provider
        .submit(admitted)
        .expect_err("released borrowed lease must be refused");
    if error.slug != "lease_not_imported" {
        return Err(format!("released borrowed lease refused with {}", error.slug).into());
    }

    provider
        .release_completion(read_token)
        .map_err(provider_error)?;
    provider
        .release_completion(write_token)
        .map_err(provider_error)?;
    provider
        .release_pipeline(&pipeline)
        .map_err(provider_error)?;
    println!(
        "PASS provider_borrowed_lease lease=98 alignment={alignment} copy_in=live copy_out=in_place retired=true refusal=lease_not_imported"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn borrowed_lease_trace(
    epoch: DeviceEpoch,
    pipeline: &CompiledComputePipeline,
    lease_id: LeaseId,
    lease_access: BufferAccess,
    owned: BufferSource,
    lease_view_id: u64,
    owned_view_id: u64,
) -> ComputeTrace {
    let owned_access = match lease_access {
        BufferAccess::Read => BufferAccess::Write,
        BufferAccess::Write => BufferAccess::Read,
        other => panic!("borrowed lease fixture cannot use {other:?}"),
    };
    let lease_view = |metal_binding| BufferView {
        view_id: ViewId::new(lease_view_id),
        metal_binding,
        allocation_id: AllocationId::new(298),
        offset: 0,
        length: 4,
        access: lease_access,
        attribute_stride: None,
        source: BufferSource::BorrowedNoCopy(lease_id),
    };
    let owned_view = |metal_binding| BufferView {
        view_id: ViewId::new(owned_view_id),
        metal_binding,
        allocation_id: AllocationId::new(299),
        offset: 0,
        length: 4,
        access: owned_access,
        attribute_stride: None,
        source: owned.clone(),
    };
    let buffers = match lease_access {
        BufferAccess::Read => vec![lease_view(0), owned_view(1)],
        BufferAccess::Write => vec![owned_view(0), lease_view(1)],
        other => panic!("borrowed lease fixture cannot use {other:?}"),
    };
    ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: epoch,
        operation_id: OperationId::new(lease_view_id),
        pipelines: vec![pipeline.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![ComputePass {
            pipeline: pipeline.pipeline_id,
            buffers,
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            },
        }],
        completion_policy: CompletionPolicy::HostReadback,
    }
}

fn borrowed_lease_resources(
    epoch: DeviceEpoch,
    reservation: LeaseReservation,
) -> Result<ResourceTableSnapshot, Box<dyn Error>> {
    let mut resources = ResourceTableSnapshot::new();
    resources.insert_allocation(AllocationRecord {
        allocation_id: AllocationId::new(298),
        owner_epoch: epoch,
        size: 64,
    })?;
    resources.insert_allocation(AllocationRecord {
        allocation_id: AllocationId::new(299),
        owner_epoch: epoch,
        size: 32,
    })?;
    resources.insert_lease(reservation)?;
    Ok(resources)
}

/// Page-aligned owner allocation for `VK_EXT_external_memory_host` imports.
struct AlignedBuffer {
    pointer: std::ptr::NonNull<u8>,
    layout: std::alloc::Layout,
}

impl AlignedBuffer {
    fn new(len: usize, alignment: usize) -> Result<Self, Box<dyn Error>> {
        let layout = std::alloc::Layout::from_size_align(len, alignment)?;
        let pointer = unsafe { std::alloc::alloc(layout) };
        let pointer = std::ptr::NonNull::new(pointer).ok_or("aligned allocation failed")?;
        Ok(Self { pointer, layout })
    }

    fn as_ptr(&self) -> *mut u8 {
        self.pointer.as_ptr()
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.layout.size()) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.layout.size()) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

fn run_indexed_and_refusals(
    provider: &VulkanComputeProvider,
    peer: &VulkanComputeProvider,
    device: &Device,
) -> Result<(), Box<dyn Error>> {
    let library = device.new_library_with_air(include_str!(
        "../shaders/kernel_dispatch_threads_boundary_barrier.ll"
    ))?;
    let function = library.function("kernel_dispatch_threads_boundary_barrier")?;
    let pipeline = provider
        .compile_pipeline(
            &function,
            SemanticDigest::new("metal-smoke-fixture-v1", b"indexed_boundary".to_vec())?,
        )
        .map_err(provider_error)?;
    let trace = make_trace(
        &pipeline,
        4,
        Dispatch {
            kind: DispatchKind::ThreadsExact,
            grid: [10, 3, 1],
            threads_per_threadgroup: [8, 2, 1],
        },
        vec![(0, 32, vec![0xaa; 30 * size_of::<u32>()])],
    )?;
    let result = submit_and_wait(provider, &trace)?;
    let expected = indexed_boundary_golden();
    check_writeback(&trace, &result, 0, &expected)?;
    if execute_indexed_boundary_dispatch(device)? != expected {
        return Err("indexed snapshot executor disagrees with the provider golden".into());
    }
    println!(
        "PASS provider_indexed_boundary_dispatch words=30 regions=4 writeback_offset=32 snapshot_parity=exact"
    );

    // Admission can verify an internally consistent proof but cannot establish
    // that it belongs to the compiled pipeline. The provider must check that.
    let mut forged = trace.clone();
    let FootprintProof::Affine { accesses } =
        &mut forged.pipelines[0].contract.buffer_bindings[0].footprint
    else {
        return Err("indexed provider fixture must carry an affine footprint proof".into());
    };
    for access in accesses {
        access.base_offset = 0;
        access.access_size = 1;
        access.terms.clear();
    }
    if forged.pipelines[0].contract == trace.pipelines[0].contract {
        return Err("forged fixture failed to change the pipeline contract".into());
    }
    let admitted = provider
        .capabilities()
        .validate_trace(forged.clone(), resources_for_trace(&forged)?)
        .map_err(provider_error)?;
    expect_refusal(provider.submit(admitted), "pipeline_contract_mismatch")?;
    println!("PASS provider_refusal slug=pipeline_contract_mismatch admitted_forgery=true");

    let admitted = peer
        .capabilities()
        .validate_trace(trace.clone(), resources_for_trace(&trace)?)
        .map_err(provider_error)?;
    expect_refusal(peer.submit(admitted), "device_epoch_mismatch")?;
    let token = result
        .completion
        .token()
        .ok_or("completed result has no token")?;
    expect_refusal(peer.wait(token, Duration::ZERO), "device_epoch_mismatch")?;
    println!("PASS provider_refusal slug=device_epoch_mismatch shared_executor=trace_and_token");
    // Compilation and release must carry owner identity through the shared API.
    expect_refusal(
        PipelineProvider::release_pipeline(peer, &pipeline),
        "device_epoch_mismatch",
    )?;
    let mut changed_pipeline = pipeline.clone();
    changed_pipeline.function.entry_name = "forged".into();
    expect_refusal(
        PipelineProvider::release_pipeline(provider, &changed_pipeline),
        "pipeline_identity_mismatch",
    )?;
    expect_refusal(
        PipelineProvider::compile(
            provider,
            PipelineCompileRequest {
                entry_name: "unsupported_msl".into(),
                logical_digest: SemanticDigest::new("fixture", vec![1])?,
                source: ShaderSource::MetalSource("kernel void unsupported_msl() {}".into()),
            },
        ),
        "shader_source_unsupported",
    )?;
    println!("PASS shared_compile_refusals foreign_release=checked metadata=checked msl=refused");

    release_case(provider, &pipeline, &result)?;
    expect_unknown_completion(provider.wait(token, Duration::ZERO), token)?;
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources_for_trace(&trace)?)
        .map_err(provider_error)?;
    expect_refusal(provider.submit(admitted), "unknown_pipeline")?;
    println!("PASS provider_release completion=unknown_completion pipeline=unknown_pipeline");
    Ok(())
}

fn make_trace(
    pipeline: &CompiledComputePipeline,
    operation: u64,
    dispatch: Dispatch,
    bindings: Vec<(u32, u64, Vec<u8>)>,
) -> Result<ComputeTrace, Box<dyn Error>> {
    let mut buffers = Vec::with_capacity(bindings.len());
    for (index, offset, bytes) in bindings {
        let access = pipeline
            .contract
            .buffer_bindings
            .iter()
            .find(|binding| binding.metal_binding == index)
            .ok_or("fixture binding is missing from pipeline reflection")?
            .access;
        buffers.push(BufferView {
            view_id: ViewId::new(200 + u64::from(index)),
            metal_binding: index,
            allocation_id: AllocationId::new(100 + u64::from(index)),
            offset,
            length: u64::try_from(bytes.len())?,
            access,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(bytes),
        });
    }
    Ok(ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: pipeline.device_epoch,
        operation_id: OperationId::new(operation),
        pipelines: vec![pipeline.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![ComputePass {
            pipeline: pipeline.pipeline_id,
            buffers,
            dispatch,
        }],
        completion_policy: CompletionPolicy::HostReadback,
    })
}

fn resources_for_trace(trace: &ComputeTrace) -> Result<ResourceTableSnapshot, Box<dyn Error>> {
    let mut resources = ResourceTableSnapshot::new();
    for view in &trace.passes[0].buffers {
        resources.insert_allocation(AllocationRecord {
            allocation_id: view.allocation_id,
            owner_epoch: trace.device_epoch,
            size: view.offset + view.length + 8,
        })?;
    }
    Ok(resources)
}

fn submit_and_wait(
    provider: &VulkanComputeProvider,
    trace: &ComputeTrace,
) -> Result<ProviderSubmission, Box<dyn Error>> {
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources_for_trace(trace)?)
        .map_err(provider_error)?;
    let result = provider.submit(admitted).map_err(provider_error)?;
    result.validate_for_trace(trace)?;
    let CompletionDisposition::CompletedVisible { token } = result.completion else {
        return Err(format!("host readback did not complete: {:?}", result.completion).into());
    };
    let waited = provider
        .wait(token, Duration::ZERO)
        .map_err(provider_error)?;
    if waited != (CompletionDisposition::CompletedVisible { token }) {
        return Err(format!("completed token did not stay visible: {waited:?}").into());
    }
    Ok(result)
}

fn check_writeback(
    trace: &ComputeTrace,
    result: &ProviderSubmission,
    binding: u32,
    expected: &[u8],
) -> Result<(), Box<dyn Error>> {
    let view = trace.passes[0]
        .buffers
        .iter()
        .find(|view| view.metal_binding == binding)
        .ok_or("output binding is missing from fixture")?;
    let [writeback] = result.writebacks.as_slice() else {
        return Err("provider fixture requires exactly one writable view".into());
    };
    if writeback.allocation_id != view.allocation_id
        || writeback.view_id != view.view_id
        || writeback.offset != view.offset
        || writeback.bytes != expected
    {
        return Err(
            format!("provider writeback did not match binding {binding}: {writeback:?}").into(),
        );
    }
    // Land the returned bytes using their allocation-relative offset and
    // compare the entire backing, including guards outside the view.
    let start = usize::try_from(view.offset)?;
    let end = start + expected.len();
    let mut actual_allocation = vec![0x5a; end + 8];
    let mut expected_allocation = actual_allocation.clone();
    expected_allocation[start..end].copy_from_slice(expected);
    let write_start = usize::try_from(writeback.offset)?;
    actual_allocation[write_start..write_start + writeback.bytes.len()]
        .copy_from_slice(&writeback.bytes);
    if actual_allocation != expected_allocation {
        return Err("provider writeback changed bytes outside its allocation view".into());
    }
    Ok(())
}

fn release_case(
    provider: &VulkanComputeProvider,
    pipeline: &CompiledComputePipeline,
    result: &ProviderSubmission,
) -> Result<(), Box<dyn Error>> {
    provider
        .release_completion(
            result
                .completion
                .token()
                .ok_or("completed result has no token")?,
        )
        .map_err(provider_error)?;
    provider
        .release_pipeline(pipeline)
        .map_err(provider_error)?;
    Ok(())
}

fn expect_unknown_completion(
    result: Result<CompletionDisposition, ProviderError>,
    token: CompletionToken,
) -> Result<(), Box<dyn Error>> {
    match result {
        Err(error)
            if error.slug == "unknown_completion"
                && error.completion
                    == (CompletionDisposition::SubmittedUnknown { token: Some(token) }) =>
        {
            Ok(())
        }
        other => {
            Err(format!("unknown token should preserve uncertain completion: {other:?}").into())
        }
    }
}

fn expect_refusal<T>(result: Result<T, ProviderError>, slug: &str) -> Result<(), Box<dyn Error>> {
    match result {
        Err(error)
            if error.slug == slug && error.completion == CompletionDisposition::NotSubmitted =>
        {
            Ok(())
        }
        Err(error) => Err(format!("expected pre-submit refusal {slug}, got {error:?}").into()),
        Ok(_) => Err(format!("provider accepted fixture requiring refusal {slug}").into()),
    }
}
