//! Shared completion slot for providers that defer GPU results.
//!
//! A synchronous provider can publish a terminal record from `submit`. An
//! asynchronous provider inserts a running record and fills it from a worker
//! thread or device completion handler. `wait` reports non-terminal timeouts;
//! `readback` returns host-visible writebacks only after completion.

use crate::provider::{
    BufferWriteback, CompletionDisposition, CompletionReadback, CompletionToken, ProviderError,
    ProviderErrorClass, ProviderPhase, Retryability,
};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

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
}

impl CompletionRecord {
    /// Create a record whose device work has not reached a terminal state.
    pub fn running() -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            state: Mutex::new(CompletionState::Running),
            ready: Condvar::new(),
        })
    }

    /// Create a record that is already completed with host-visible writebacks.
    pub fn completed(writebacks: Vec<BufferWriteback>) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            state: Mutex::new(CompletionState::Completed(writebacks)),
            ready: Condvar::new(),
        })
    }

    /// Create a record that is already failed.
    pub fn failed(error: ProviderError) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            state: Mutex::new(CompletionState::Failed(error)),
            ready: Condvar::new(),
        })
    }

    /// Record successful completion. Ignored if a terminal state already won.
    pub fn complete(&self, writebacks: Vec<BufferWriteback>) {
        match self.state.lock() {
            Ok(mut state) => {
                if matches!(*state, CompletionState::Running) {
                    *state = CompletionState::Completed(writebacks);
                }
            }
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                if matches!(*state, CompletionState::Running) {
                    *state = CompletionState::Completed(writebacks);
                }
            }
        }
        self.ready.notify_all();
    }

    /// Record a provider failure. Ignored if a terminal state already won.
    pub fn fail(&self, error: ProviderError) {
        match self.state.lock() {
            Ok(mut state) => {
                if matches!(*state, CompletionState::Running) {
                    *state = CompletionState::Failed(error);
                }
            }
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                if matches!(*state, CompletionState::Running) {
                    *state = CompletionState::Failed(error);
                }
            }
        }
        self.ready.notify_all();
    }

    /// Record host-requested cancellation. Ignored if a terminal state already
    /// won. Cancellation abandons the observation of device work; it is not
    /// evidence that the GPU retired the submission.
    pub fn cancel(&self) {
        match self.state.lock() {
            Ok(mut state) => {
                if matches!(*state, CompletionState::Running) {
                    *state = CompletionState::Cancelled;
                }
            }
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                if matches!(*state, CompletionState::Running) {
                    *state = CompletionState::Cancelled;
                }
            }
        }
        self.ready.notify_all();
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
}
