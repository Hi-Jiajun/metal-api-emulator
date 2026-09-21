//! The provider's own arm for the zero-fill declaration (statement economy
//! W2-A, `research/docs/23` §121).
//!
//! A trace states a zero-fill declaration either by decoding the arm off the
//! wire (`metal-api-ipc`'s buffer-source tag, which is how the render rail's
//! two producers hand it over) or by building it in process. Both arrive at
//! [`crate::BindingBytes`]-shaped uploads the same way, and both are
//! **materialized here**: the provider writes `length` zero bytes into its own
//! upload at the view's own window, exactly where the `OwnedBytes` payload the
//! arm replaced would have been copied in. Nothing downstream can tell the two
//! encodings apart, which is the whole claim of the arm.
//!
//! # The switch
//!
//! `METAL_API_VULKAN_ZERO_FILL_DECLARATIONS` lets a *round* take the arm's
//! reading for every all-zero `OwnedBytes` declaration it is handed, so the
//! arm's materialization is exercised over a corpus that never states it.
//! **Off by default**; off, a payload is uploaded exactly as it always was and
//! this module is one relaxed load per declaration. On, a payload whose bytes
//! are all zero takes the arm's path instead — the same device-visible bytes
//! by construction, which is what makes the two Lavapipe smoke arms' 141
//! captures a statement about the materialization rather than about the
//! fixture.
//!
//! A payload with any nonzero byte never takes the arm on either setting: the
//! arm states one content, and a declaration that means others uploads them.

use std::sync::atomic::{AtomicU8, Ordering};

/// The environment spelling the round's launcher writes.
const SWITCH: &str = "METAL_API_VULKAN_ZERO_FILL_DECLARATIONS";

const UNSET: u8 = 0;
const ON: u8 = 1;
const OFF: u8 = 2;

static ARM: AtomicU8 = AtomicU8::new(UNSET);

/// Whether this process takes the zero-fill arm's reading for an all-zero
/// owned payload.
pub(crate) fn enabled() -> bool {
    match ARM.load(Ordering::Relaxed) {
        ON => true,
        OFF => false,
        _ => {
            let on = parse(std::env::var(SWITCH).ok().as_deref());
            ARM.store(if on { ON } else { OFF }, Ordering::Relaxed);
            on
        }
    }
}

/// Pin the arm for the rest of the process, overriding [`SWITCH`]. Tests use
/// this the way a round uses the launcher: one arm per process.
#[cfg(test)]
pub(crate) fn set_enabled(on: bool) {
    ARM.store(if on { ON } else { OFF }, Ordering::Relaxed);
}

/// Whether `bytes` is a payload the arm may stand for: every byte zero. The
/// predicate is the arm's own content rule, so it is stated once here rather
/// than at each take site.
pub(crate) fn stands_for(bytes: &[u8]) -> bool {
    bytes.iter().all(|byte| *byte == 0)
}

/// The bytes the arm materializes for a declaration of `length` bytes.
pub(crate) fn materialize(length: u64) -> Option<Vec<u8>> {
    usize::try_from(length).ok().map(|length| vec![0_u8; length])
}

/// The switch's own parser, apart from the process-global it caches into so a
/// unit test can read every spelling without a switch it cannot put back.
fn parse(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "on" | "true" | "yes")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_reads_the_spellings_a_launcher_writes() {
        for on in ["1", "on", "ON", " true ", "yes"] {
            assert!(parse(Some(on)), "{on:?} asks for the arm");
        }
        for off in ["0", "off", "false", "no", ""] {
            assert!(!parse(Some(off)), "{off:?} does not ask for the arm");
        }
        assert!(!parse(None), "unset does not ask for the arm");
    }

    #[test]
    fn only_an_all_zero_payload_is_what_the_arm_stands_for() {
        assert!(stands_for(&[]));
        assert!(stands_for(&[0, 0, 0, 0]));
        assert!(!stands_for(&[0, 0, 7, 0]));
        assert!(!stands_for(&[0xff]));
    }

    #[test]
    fn materializing_is_the_declared_length_and_nothing_else() {
        assert_eq!(materialize(0), Some(Vec::new()));
        assert_eq!(materialize(4), Some(vec![0, 0, 0, 0]));
        assert_eq!(materialize(u64::MAX), None);
    }
}
