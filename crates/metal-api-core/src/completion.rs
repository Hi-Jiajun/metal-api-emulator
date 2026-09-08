//! Shared completion slot for providers that defer GPU results.
//!
//! A synchronous provider can publish a terminal record from `submit`. An
//! asynchronous provider inserts a running record and fills it from a worker
//! thread or device completion handler. `wait` reports non-terminal timeouts;
//! `readback` returns host-visible writebacks only after completion.
//!
//! The [`wire`] submodule carries the same terminal semantics across a process
//! boundary as transport-independent notifications.

use crate::provider::{
    BufferWriteback, CompletionDisposition, CompletionReadback, CompletionToken, ProviderError,
    ProviderErrorClass, ProviderPhase, Retryability,
};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

pub mod wire;

enum CompletionState {
    Running,
    Completed(Vec<BufferWriteback>),
    Cancelled,
    Failed(ProviderError),
}

/// A monotonic observation bound shared by deferred-completion providers.
///
/// A provider starts one deadline when it records a `Submitted` submission.
/// `wait` clamps the caller's timeout to `remaining`; once `expired` is true,
/// the provider must publish a terminal unknown completion instead of a
/// non-terminal `TimedOut`. The deadline measures host observation time, not
/// device execution time.
#[derive(Clone, Copy, Debug)]
pub struct ObservationDeadline {
    started: Instant,
    limit: Duration,
}

impl ObservationDeadline {
    pub fn new(limit: Duration) -> Self {
        Self {
            started: Instant::now(),
            limit,
        }
    }

    pub fn limit(&self) -> Duration {
        self.limit
    }

    /// Time left before the provider must report unknown completion.
    pub fn remaining(&self) -> Option<Duration> {
        self.limit.checked_sub(self.started.elapsed())
    }

    pub fn expired(&self) -> bool {
        self.remaining().is_none()
    }

    /// Bound a caller timeout by the remaining observation window.
    pub fn clamp(&self, requested: Duration) -> Duration {
        requested.min(self.remaining().unwrap_or(Duration::ZERO))
    }
}

/// Terminal transition observed by a [`CompletionRecord`].
///
/// The value mirrors the disposition `wait` would report, except that a
/// timeout and `Submitted` are not terminal and therefore never observed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompletionTerminal {
    /// Host-visible writebacks are ready.
    CompletedVisible,
    /// The host cancelled observation before a result landed.
    Cancelled,
    /// The provider observed a refusal.
    Failed(ProviderError),
    /// The provider can no longer observe the submission.
    SubmittedUnknown,
}

/// Observer of the first terminal transition of a [`CompletionRecord`].
///
/// The record calls this exactly once, after the state change, without holding
/// the record lock. An implementation must not block the provider for long and
/// must not call back into the same record. A provider uses it to publish the
/// same terminal transition it stores in-process; see
/// [`wire::CompletionOutbox`].
pub trait CompletionObserver: Send + Sync {
    fn observe_terminal(&self, token: CompletionToken, terminal: CompletionTerminal);
}

struct TerminalObserver {
    token: CompletionToken,
    observer: Arc<dyn CompletionObserver>,
}

/// Shared slot between a submit path and later `wait`/`readback` calls.
///
/// A synchronous provider fills the slot before returning `CompletedVisible`.
/// An asynchronous provider inserts `Running`, lets a worker or completion
/// handler fill it, and returns `Submitted` immediately. Removing the slot from
/// the provider map does not cancel in-flight work; the writer's `Arc` keeps
/// the record and its GPU resources alive until execution finishes.
///
/// The first terminal transition wins. A late completion cannot overwrite a
/// failure recorded by a deadline or by the device.
pub struct CompletionRecord {
    state: Mutex<CompletionState>,
    ready: Condvar,
    terminal_observer: Option<TerminalObserver>,
}

impl CompletionRecord {
    /// Create a record whose device work has not reached a terminal state.
    pub fn running() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(CompletionState::Running),
            ready: Condvar::new(),
            terminal_observer: None,
        })
    }

    /// Create a running record that publishes its first terminal transition
    /// through `observer`.
    pub fn running_with_observer(
        token: CompletionToken,
        observer: Arc<dyn CompletionObserver>,
    ) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(CompletionState::Running),
            ready: Condvar::new(),
            terminal_observer: Some(TerminalObserver { token, observer }),
        })
    }

    /// Create a record that is already completed with host-visible writebacks.
    pub fn completed(writebacks: Vec<BufferWriteback>) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(CompletionState::Completed(writebacks)),
            ready: Condvar::new(),
            terminal_observer: None,
        })
    }

    /// Create a record that is already failed.
    pub fn failed(error: ProviderError) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(CompletionState::Failed(error)),
            ready: Condvar::new(),
            terminal_observer: None,
        })
    }

    /// Record successful completion. Ignored if a terminal state already won.
    pub fn complete(&self, writebacks: Vec<BufferWriteback>) {
        let mut changed = false;
        match self.state.lock() {
            Ok(mut state) => {
                if matches!(*state, CompletionState::Running) {
                    *state = CompletionState::Completed(writebacks);
                    changed = true;
                }
            }
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                if matches!(*state, CompletionState::Running) {
                    *state = CompletionState::Completed(writebacks);
                    changed = true;
                }
            }
        }
        self.ready.notify_all();
        if changed {
            self.notify_terminal(CompletionTerminal::CompletedVisible);
        }
    }

    /// Record a provider failure. Ignored if a terminal state already won.
    pub fn fail(&self, error: ProviderError) {
        let terminal = match &error.completion {
            CompletionDisposition::SubmittedUnknown { .. } => CompletionTerminal::SubmittedUnknown,
            _ => CompletionTerminal::Failed(error.clone()),
        };
        let mut changed = false;
        match self.state.lock() {
            Ok(mut state) => {
                if matches!(*state, CompletionState::Running) {
                    *state = CompletionState::Failed(error);
                    changed = true;
                }
            }
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                if matches!(*state, CompletionState::Running) {
                    *state = CompletionState::Failed(error);
                    changed = true;
                }
            }
        }
        self.ready.notify_all();
        if changed {
            self.notify_terminal(terminal);
        }
    }

    /// Record host-requested cancellation. Ignored if a terminal state already
    /// won. Cancellation abandons the observation of device work; it is not
    /// evidence that the GPU retired the submission.
    pub fn cancel(&self) {
        let mut changed = false;
        match self.state.lock() {
            Ok(mut state) => {
                if matches!(*state, CompletionState::Running) {
                    *state = CompletionState::Cancelled;
                    changed = true;
                }
            }
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                if matches!(*state, CompletionState::Running) {
                    *state = CompletionState::Cancelled;
                    changed = true;
                }
            }
        }
        self.ready.notify_all();
        if changed {
            self.notify_terminal(CompletionTerminal::Cancelled);
        }
    }

    fn notify_terminal(&self, terminal: CompletionTerminal) {
        if let Some(entry) = &self.terminal_observer {
            entry.observer.observe_terminal(entry.token, terminal);
        }
    }

    /// Whether the record has not yet observed a terminal state.
    pub fn is_running(&self) -> bool {
        match self.state.lock() {
            Ok(state) => matches!(*state, CompletionState::Running),
            Err(poisoned) => matches!(*poisoned.into_inner(), CompletionState::Running),
        }
    }

    /// Observe this record until it reaches a terminal state or the caller's
    /// timeout expires. A timeout is a non-terminal disposition, not an error.
    pub fn wait(
        &self,
        token: CompletionToken,
        timeout: Duration,
    ) -> Result<CompletionDisposition, ProviderError> {
        let deadline = Instant::now().checked_add(timeout);
        let mut state = self.state.lock().map_err(|_| record_poisoned())?;
        loop {
            match &*state {
                CompletionState::Running => {
                    let Some(deadline) = deadline else {
                        return Ok(CompletionDisposition::TimedOut { token });
                    };
                    let now = Instant::now();
                    if now >= deadline {
                        return Ok(CompletionDisposition::TimedOut { token });
                    }
                    let (next, result) = self
                        .ready
                        .wait_timeout(state, deadline - now)
                        .map_err(|_| record_poisoned())?;
                    state = next;
                    if result.timed_out() && matches!(&*state, CompletionState::Running) {
                        return Ok(CompletionDisposition::TimedOut { token });
                    }
                }
                CompletionState::Completed(_) => {
                    return Ok(CompletionDisposition::CompletedVisible { token })
                }
                CompletionState::Cancelled => {
                    return Ok(CompletionDisposition::Cancelled { token })
                }
                CompletionState::Failed(error) => return Err(error.clone()),
            }
        }
    }

    /// Return writebacks for a token that already completed. Reading before a
    /// terminal observation is refused with a non-terminal `Submitted` error.
    pub fn readback(&self, token: CompletionToken) -> Result<CompletionReadback, ProviderError> {
        let state = self.state.lock().map_err(|_| record_poisoned())?;
        match &*state {
            CompletionState::Completed(writebacks) => Ok(CompletionReadback {
                completion: CompletionDisposition::CompletedVisible { token },
                writebacks: writebacks.clone(),
            }),
            CompletionState::Cancelled => Err(ProviderError::new(
                ProviderPhase::Readback,
                ProviderErrorClass::Execute,
                "completion_cancelled",
            )
            .expect("non-empty completion record error slug")
            .with_completion(CompletionDisposition::Cancelled { token })),
            CompletionState::Failed(error) => Err(error.clone()),
            CompletionState::Running => Err(ProviderError::new(
                ProviderPhase::Readback,
                ProviderErrorClass::Resource,
                "completion_not_ready",
            )
            .expect("non-empty completion record error slug")
            .with_completion(CompletionDisposition::Submitted { token })),
        }
    }
}

fn record_poisoned() -> ProviderError {
    let mut error = ProviderError::new(
        ProviderPhase::Wait,
        ProviderErrorClass::Internal,
        "completion_record_poisoned",
    )
    .expect("non-empty completion record error slug");
    error.retryability = Retryability::Never;
    error
}

/// Bounded allowance for submissions whose completion can no longer be observed.
///
/// The budget is provider-scoped and monotonic. Abandoned resources cannot be
/// safely returned to the device, so the ledger never refunds bytes; once the
/// configured limit is reached the provider must refuse new work and be
/// recreated. `max_submissions` is the abandoned count at which the budget is
/// exhausted (a value of 1 fails closed on the first abandonment).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AbandonmentBudget {
    max_submissions: u64,
    max_bytes: u64,
}

/// Result of recording one abandoned submission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AbandonmentOutcome {
    Admitted,
    Exhausted,
}

impl AbandonmentBudget {
    pub const fn new(max_submissions: u64, max_bytes: u64) -> Self {
        Self {
            max_submissions,
            max_bytes,
        }
    }

    pub const fn max_submissions(self) -> u64 {
        self.max_submissions
    }

    pub const fn max_bytes(self) -> u64 {
        self.max_bytes
    }
}

/// Monotonic accounting for abandoned submissions in one provider instance.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AbandonmentLedger {
    submissions: u64,
    bytes: u64,
}

impl AbandonmentLedger {
    /// Record one abandonment and report whether the budget is still intact.
    pub fn record(&mut self, budget: AbandonmentBudget, bytes: u64) -> AbandonmentOutcome {
        self.submissions = self.submissions.saturating_add(1);
        self.bytes = self.bytes.saturating_add(bytes);
        if self.submissions >= budget.max_submissions || self.bytes >= budget.max_bytes {
            AbandonmentOutcome::Exhausted
        } else {
            AbandonmentOutcome::Admitted
        }
    }

    pub const fn submissions(self) -> u64 {
        self.submissions
    }

    pub const fn bytes(self) -> u64 {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{AllocationId, DeviceEpoch, SubmissionId, ViewId};

    fn completion_token() -> CompletionToken {
        CompletionToken {
            device_epoch: DeviceEpoch::new(1),
            submission_id: SubmissionId::new(2),
        }
    }

    fn completion_writeback() -> BufferWriteback {
        BufferWriteback {
            view_id: ViewId::new(3),
            allocation_id: AllocationId::new(4),
            offset: 0,
            bytes: vec![5; 4],
        }
    }

    #[test]
    fn reports_running_timeout_then_completed_readback() {
        let token = completion_token();
        let record = CompletionRecord::running();
        assert_eq!(
            record.wait(token, Duration::ZERO).unwrap(),
            CompletionDisposition::TimedOut { token }
        );
        assert!(
            matches!(record.readback(token), Err(error) if error.slug == "completion_not_ready")
        );
        record.complete(vec![completion_writeback()]);
        assert_eq!(
            record.wait(token, Duration::ZERO).unwrap(),
            CompletionDisposition::CompletedVisible { token }
        );
        let readback = record.readback(token).unwrap();
        assert_eq!(
            readback.completion,
            CompletionDisposition::CompletedVisible { token }
        );
        assert_eq!(readback.writebacks, vec![completion_writeback()]);
    }

    #[test]
    fn waiter_wakes_on_worker_completion() {
        let token = completion_token();
        let record = CompletionRecord::running();
        let worker = std::sync::Arc::clone(&record);
        let handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(10));
            worker.complete(vec![completion_writeback()]);
        });
        assert_eq!(
            record.wait(token, Duration::from_secs(5)).unwrap(),
            CompletionDisposition::CompletedVisible { token }
        );
        handle.join().unwrap();
    }

    #[test]
    fn propagates_worker_failure() {
        let token = completion_token();
        let record = CompletionRecord::running();
        let failure = ProviderError::new(
            ProviderPhase::Wait,
            ProviderErrorClass::Execute,
            "synthetic_worker_failure",
        )
        .unwrap();
        record.fail(failure.clone());
        assert_eq!(record.wait(token, Duration::ZERO).unwrap_err(), failure);
        assert_eq!(record.readback(token).unwrap_err(), failure);
    }

    #[test]
    fn first_terminal_transition_wins() {
        let token = completion_token();
        let record = CompletionRecord::running();
        let failure = ProviderError::new(
            ProviderPhase::Wait,
            ProviderErrorClass::Execute,
            "deadline_unknown",
        )
        .unwrap();
        record.fail(failure.clone());
        record.complete(vec![completion_writeback()]);
        assert_eq!(record.wait(token, Duration::ZERO).unwrap_err(), failure);
        assert_eq!(record.readback(token).unwrap_err(), failure);
    }

    #[test]
    fn cancellation_is_a_terminal_observation() {
        let token = completion_token();
        let record = CompletionRecord::running();
        record.cancel();
        assert!(!record.is_running());
        assert_eq!(
            record.wait(token, Duration::ZERO).unwrap(),
            CompletionDisposition::Cancelled { token }
        );
        let error = record.readback(token).unwrap_err();
        assert_eq!(error.slug, "completion_cancelled");
        assert_eq!(error.completion, CompletionDisposition::Cancelled { token });
        // A late device completion cannot resurrect cancelled work.
        record.complete(vec![completion_writeback()]);
        assert_eq!(
            record.wait(token, Duration::ZERO).unwrap(),
            CompletionDisposition::Cancelled { token }
        );
    }

    #[test]
    fn observation_deadline_clamps_to_the_smaller_bound() {
        let deadline = ObservationDeadline::new(Duration::from_secs(30));
        assert_eq!(
            deadline.clamp(Duration::from_secs(1)),
            Duration::from_secs(1)
        );
        assert!(deadline.clamp(Duration::from_secs(60)) <= Duration::from_secs(30));
        assert!(!deadline.expired());
        assert_eq!(deadline.limit(), Duration::from_secs(30));
    }

    #[test]
    fn expired_observation_deadline_never_grants_a_retry_window() {
        let deadline = ObservationDeadline::new(Duration::ZERO);
        assert!(deadline.expired());
        assert_eq!(deadline.remaining(), None);
        assert_eq!(deadline.clamp(Duration::from_secs(5)), Duration::ZERO);
    }

    #[test]
    fn abandonment_budget_exhausts_at_the_configured_submission_count() {
        let budget = AbandonmentBudget::new(3, 1_000);
        let mut ledger = AbandonmentLedger::default();
        assert_eq!(ledger.record(budget, 10), AbandonmentOutcome::Admitted);
        assert_eq!(ledger.record(budget, 10), AbandonmentOutcome::Admitted);
        assert_eq!(ledger.record(budget, 10), AbandonmentOutcome::Exhausted);
        assert_eq!(ledger.submissions(), 3);
        assert_eq!(ledger.bytes(), 30);
    }

    #[test]
    fn abandonment_budget_exhausts_on_an_oversized_submission() {
        let budget = AbandonmentBudget::new(8, 64);
        let mut ledger = AbandonmentLedger::default();
        assert_eq!(ledger.record(budget, 64), AbandonmentOutcome::Exhausted);
        assert_eq!(ledger.submissions(), 1);
        assert_eq!(ledger.bytes(), 64);
    }

    #[test]
    fn zero_abandonment_budget_fails_closed_immediately() {
        let budget = AbandonmentBudget::new(0, 0);
        let mut ledger = AbandonmentLedger::default();
        assert_eq!(ledger.record(budget, 0), AbandonmentOutcome::Exhausted);
    }

    #[derive(Default)]
    struct RecordingObserver {
        terminals: Mutex<Vec<(CompletionToken, CompletionTerminal)>>,
    }

    impl CompletionObserver for RecordingObserver {
        fn observe_terminal(&self, token: CompletionToken, terminal: CompletionTerminal) {
            self.terminals
                .lock()
                .expect("recording observer")
                .push((token, terminal));
        }
    }

    fn observed(observer: &Arc<RecordingObserver>) -> Vec<(CompletionToken, CompletionTerminal)> {
        observer
            .terminals
            .lock()
            .expect("recording observer")
            .clone()
    }

    #[test]
    fn observer_sees_the_first_terminal_transition_once() {
        let observer = Arc::new(RecordingObserver::default());
        let token = completion_token();
        let record = CompletionRecord::running_with_observer(token, observer.clone());
        assert!(record.is_running());
        record.complete(vec![completion_writeback()]);
        record.fail(record_poisoned());
        record.cancel();
        assert_eq!(
            observed(&observer),
            vec![(token, CompletionTerminal::CompletedVisible)]
        );
    }

    #[test]
    fn observer_distinguishes_submitted_unknown_from_failure() {
        let observer = Arc::new(RecordingObserver::default());
        let token = completion_token();
        let record = CompletionRecord::running_with_observer(token, observer.clone());
        let mut error = record_poisoned();
        error.completion = CompletionDisposition::SubmittedUnknown { token: Some(token) };
        record.fail(error);
        assert_eq!(
            observed(&observer),
            vec![(token, CompletionTerminal::SubmittedUnknown)]
        );
    }

    #[test]
    fn observer_reports_cancellation_without_a_later_completion() {
        let observer = Arc::new(RecordingObserver::default());
        let token = completion_token();
        let record = CompletionRecord::running_with_observer(token, observer.clone());
        record.cancel();
        record.complete(Vec::new());
        assert_eq!(
            observed(&observer),
            vec![(token, CompletionTerminal::Cancelled)]
        );
    }

    #[test]
    fn observer_is_not_notified_for_a_timeout() {
        let observer = Arc::new(RecordingObserver::default());
        let token = completion_token();
        let record = CompletionRecord::running_with_observer(token, observer.clone());
        assert_eq!(
            record.wait(token, Duration::ZERO).unwrap(),
            CompletionDisposition::TimedOut { token }
        );
        assert!(observed(&observer).is_empty());
    }

    #[test]
    fn failure_observer_carries_the_structured_error() {
        let observer = Arc::new(RecordingObserver::default());
        let token = completion_token();
        let record = CompletionRecord::running_with_observer(token, observer.clone());
        let mut error = record_poisoned();
        error.completion = CompletionDisposition::Failed { token: Some(token) };
        record.fail(error.clone());
        assert_eq!(
            observed(&observer),
            vec![(token, CompletionTerminal::Failed(error))]
        );
    }
}
