//! Default-off admission timing for one provider admission walk.
//!
//! [`ProviderCapabilities::admit`](crate::provider::ProviderCapabilities::admit)
//! is the neutral contract's whole admission: it is called **twice** on the
//! path a submission takes — once by
//! [`ProviderCapabilities::validate_trace`](crate::provider::ProviderCapabilities::validate_trace),
//! whose answer is the `ValidatedComputeTrace` the caller then reserves leases
//! against, and once by the receiving owner immediately before it encodes
//! (`metal-api-vulkan`'s `submit`, whose comment states why: a validated trace
//! may have been admitted against another snapshot).
//!
//! Outside this crate the second call is visible as one bar with no interior —
//! the reims rail's `prov_admit_validate_us_mean` brackets the first and
//! `metal-api-vulkan`'s `admit` phase brackets the second, and neither can say
//! *which* gate of the walk the microseconds are in. On the 2026-09-21 sixth
//! profile round (`r_submit_profile6`) the first was **18.9 ms/frame** in the
//! production pose — 5.6× its census-pose reading — and nothing in the reading
//! said why.
//!
//! This module splits the walk into the regions a fix would move, driven by
//! `METAL_API_CORE_ADMIT_PROFILE`:
//!
//! * unset (the default) or any value other than a truthy one: off. Every call
//!   site is a `let _bar = Bar::enter(site, Region::X)` that resolves to `None`
//!   after one relaxed load — **no clock read, no lock, no allocation**.
//! * `1` / `on` / `true` / `yes`: on. Each bar reads the monotonic clock twice
//!   and adds to a thread-local table; every `METAL_API_CORE_ADMIT_PROFILE_EVERY`
//!   (default 256) completed walks **of one route** that thread writes one line
//!   to stderr (the QEMU process's own stderr, i.e. the round's boot log):
//!
//!   ```text
//!   ADMIT route=validate n=256 refused_n=0 total_us=... named_us=...
//!   trace_validate_us=... render_passes_us=... landings_us=...
//!   render_textures_us=... pixel_samplers_us=... stage_buffers_us=...
//!   compute_textures_us=... present_us=... heap_us=... indirect_us=...
//!   limits_us=... compute_passes_us=... serial_reuse_us=... resources_us=...
//!   passes_n=... render_passes_n=... compute_passes_n=... draws_n=...
//!   landings_n=... pipelines_n=... stage_buffers_n=... textures_n=...
//!   samplers_n=... color_attachments_n=... compute_views_n=...
//!   vertex_buffers_n=... allocations_n=... leases_n=...
//!   ```
//!
//! The fourteen regions are **disjoint consecutive regions of one call**, in
//! the order `admit` runs them, so `sum(fields) <= total_us` and the difference
//! is the seam between the brackets. `named_us` is printed beside them as the
//! aggregate rather than added to the sum, exactly as `plan_settle_us` is in
//! `metal-api-vulkan::phase_profile`.
//!
//! The `*_n` fields are the shape census of the same window: the number of
//! passes, pipelines, stage buffers, textures, samplers and views the admitted
//! traces carried. They are what answers "is this pose's walk slower because it
//! is called more, or because each trace is bigger" — and they are sums over
//! the window's walks, so dividing by `n` gives the mean shape of one trace.
//!
//! Fields are **sums**, not means, so a reader can add lines together and
//! divide by the summed `n` without weighting error. µs fields carry three
//! decimals, i.e. nanosecond resolution.

use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::Instant;

/// The route a walk was reached by: the three callers of `admit` differ in what
/// the caller already knows, and a fix that removes one route's walk (an owner
/// that is re-checking the very snapshot that admitted the trace) shows up as
/// that route's line collapsing while the others stay put.
pub(crate) const SITE_COUNT: usize = 3;
/// [`ProviderCapabilities::validate_trace`], the caller-facing admission.
///
/// [`ProviderCapabilities::validate_trace`]: crate::provider::ProviderCapabilities::validate_trace
pub(crate) const SITE_VALIDATE: usize = 0;
/// The receiving owner's re-check before it encodes the submission.
pub(crate) const SITE_SUBMIT: usize = 1;
/// A direct `admit` call that is neither of the two above: the native rail, the
/// provider-object API and every test that calls the walk by name.
pub(crate) const SITE_DIRECT: usize = 2;
const SITE_NAMES: [&str; SITE_COUNT] = ["validate", "submit", "direct"];

/// One disjoint region of the walk, in the order the walk runs them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Region {
    /// `ComputeTrace::validate`: the trace's own structural rules — pipeline
    /// identity and epoch, every pass's own shape, the render entries'
    /// attachment and declaration rules, the heap/ICB payloads.
    TraceValidate = 0,
    /// `admit_render_passes`: the pipeline/pass pair rules and the capability
    /// bits the render shape asks for.
    RenderPasses,
    /// `admit_kept_frame_landings`: the landing-only entries' format and extent
    /// gates.
    Landings,
    /// `admit_render_texture_inputs`: the sampled texture gate — bit, count,
    /// format.
    RenderTextures,
    /// `admit_render_pixel_samplers`: the runtime sampler state gate.
    PixelSamplers,
    /// `admit_render_stage_buffer_inputs`: the stage buffer gate — bit, count,
    /// source, per-stage ceiling.
    StageBuffers,
    /// `admit_compute_texture_inputs`: the compute half's sampling gate.
    ComputeTextures,
    /// `admit_present_actions`: the present gate.
    Present,
    /// `admit_heap_payload`: the heap gate.
    Heap,
    /// `admit_indirect_payload`: the ICB gate.
    Indirect,
    /// The whole-trace limits: pass count, dispatch type, completion policy.
    Limits,
    /// The compute-pass walk: per-pass contract checks, the dispatch and push
    /// constant limits, the footprint proofs and the allocation/view table.
    /// A render-only trace (this rail's production shape) walks nothing here,
    /// so the bar is also the reading that says so.
    ComputePasses,
    /// `ComputeTrace::validate_serial_buffer_reuse`: the serial reuse subset.
    SerialReuse,
    /// `ResourceTableSnapshot::validate_trace`: the resource namespace walk.
    Resources,
}

const REGION_COUNT: usize = Region::Resources as usize + 1;
const REGION_NAMES: [&str; REGION_COUNT] = [
    "trace_validate",
    "render_passes",
    "landings",
    "render_textures",
    "pixel_samplers",
    "stage_buffers",
    "compute_textures",
    "present",
    "heap",
    "indirect",
    "limits",
    "compute_passes",
    "serial_reuse",
    "resources",
];

/// How many shapes the census counts, and what each slot counts.
///
/// The first twelve are sums over the window's traces of what each trace
/// carried; the last two are the resource snapshot's own size. Together they
/// are the denominator a bar reading needs: the same 18.9 ms is a different
/// finding when it is 71.8 small traces a frame than when it is 12 large ones.
const CENSUS_COUNT: usize = 14;
const CENSUS_NAMES: [&str; CENSUS_COUNT] = [
    "passes_n",
    "render_passes_n",
    "compute_passes_n",
    "draws_n",
    "landings_n",
    "pipelines_n",
    "stage_buffers_n",
    "textures_n",
    "samplers_n",
    "color_attachments_n",
    "compute_views_n",
    "vertex_buffers_n",
    "allocations_n",
    "leases_n",
];

const EVERY_DEFAULT: u64 = 256;

/// One walk's shape, as the census counts it.
#[derive(Clone, Copy, Default)]
pub(crate) struct Census {
    pub passes: u64,
    pub render_passes: u64,
    pub compute_passes: u64,
    pub draws: u64,
    pub landings: u64,
    pub pipelines: u64,
    pub stage_buffers: u64,
    pub textures: u64,
    pub samplers: u64,
    pub color_attachments: u64,
    pub compute_views: u64,
    pub vertex_buffers: u64,
    pub allocations: u64,
    pub leases: u64,
}

/// One thread's window of the profile.
struct Local {
    ns: [[u64; REGION_COUNT]; SITE_COUNT],
    total_ns: [u64; SITE_COUNT],
    calls: [u64; SITE_COUNT],
    window: [u64; SITE_COUNT],
    refused: [u64; SITE_COUNT],
    census: [[u64; CENSUS_COUNT]; SITE_COUNT],
}

impl Local {
    fn new() -> Self {
        Self {
            ns: [[0; REGION_COUNT]; SITE_COUNT],
            total_ns: [0; SITE_COUNT],
            calls: [0; SITE_COUNT],
            window: [0; SITE_COUNT],
            refused: [0; SITE_COUNT],
            census: [[0; CENSUS_COUNT]; SITE_COUNT],
        }
    }
}

thread_local! {
    static LOCAL: RefCell<Local> = RefCell::new(Local::new());
}

/// Whether the profile is on, read once from the process environment.
#[inline]
fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_enabled(
            std::env::var("METAL_API_CORE_ADMIT_PROFILE")
                .ok()
                .as_deref(),
        )
    })
}

fn every() -> u64 {
    static EVERY: OnceLock<u64> = OnceLock::new();
    *EVERY.get_or_init(|| {
        std::env::var("METAL_API_CORE_ADMIT_PROFILE_EVERY")
            .ok()
            .and_then(|value| value.trim().parse().ok())
            .filter(|value| *value > 0)
            .unwrap_or(EVERY_DEFAULT)
    })
}

/// The enable word, read exactly as a round's launcher spells it.
fn parse_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim),
        Some("1" | "on" | "ON" | "true" | "yes")
    )
}

/// The enclosing bar of one walk: minted first, dropped last, and the thing
/// that counts the window and emits it.
pub(crate) struct Window {
    site: usize,
    started: Instant,
    refused: bool,
    census: Census,
}

impl Window {
    #[inline]
    pub(crate) fn enter(site: usize) -> Option<Self> {
        if !enabled() {
            return None;
        }
        Some(Self {
            site,
            started: Instant::now(),
            refused: false,
            census: Census::default(),
        })
    }

    /// Charge the walk's own shape census. Called once per walk, after the
    /// trace has been read (the counts are the walk's inputs, not its output).
    #[inline]
    pub(crate) fn note_census(&mut self, census: Census) {
        self.census = census;
    }

    /// Record that the walk refused the trace. Reported beside the timings
    /// rather than derived from them: a pose whose walk is slow *because* it
    /// is rejecting and being retried has a different fix from one whose walk
    /// is slow on traces it admits.
    #[inline]
    pub(crate) fn note_refused(&mut self) {
        self.refused = true;
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        let elapsed = self.started.elapsed().as_nanos() as u64;
        let site = self.site;
        let census = self.census;
        let refused = self.refused;
        let emit = LOCAL.with(|local| {
            let mut local = local.borrow_mut();
            local.total_ns[site] += elapsed;
            local.calls[site] += 1;
            if refused {
                local.refused[site] += 1;
            }
            local.census[site][0] += census.passes;
            local.census[site][1] += census.render_passes;
            local.census[site][2] += census.compute_passes;
            local.census[site][3] += census.draws;
            local.census[site][4] += census.landings;
            local.census[site][5] += census.pipelines;
            local.census[site][6] += census.stage_buffers;
            local.census[site][7] += census.textures;
            local.census[site][8] += census.samplers;
            local.census[site][9] += census.color_attachments;
            local.census[site][10] += census.compute_views;
            local.census[site][11] += census.vertex_buffers;
            local.census[site][12] += census.allocations;
            local.census[site][13] += census.leases;
            local.window[site] += 1;
            local.window[site] >= every()
        });
        if emit {
            emit_line(site);
        }
    }
}

/// One region's bar, banked when it drops.
///
/// `None` when the profile is off, which keeps every call site a
/// `let _bar = Bar::enter(site, Region::X)` with no branch in it: the switch is
/// read once here rather than at each bracket.
pub(crate) struct Bar {
    site: usize,
    region: usize,
    started: Instant,
}

impl Bar {
    #[inline]
    pub(crate) fn enter(site: usize, region: Region) -> Option<Self> {
        if !enabled() {
            return None;
        }
        Some(Self {
            site,
            region: region as usize,
            started: Instant::now(),
        })
    }
}

impl Drop for Bar {
    fn drop(&mut self) {
        let elapsed = self.started.elapsed().as_nanos() as u64;
        let (site, region) = (self.site, self.region);
        LOCAL.with(|local| local.borrow_mut().ns[site][region] += elapsed);
    }
}

/// One window of one route, taken out of the thread's table.
struct Taken {
    ns: [u64; REGION_COUNT],
    census: [u64; CENSUS_COUNT],
    total_ns: u64,
    n: u64,
    refused_n: u64,
}

/// Take one route's window out of the thread's table, leaving it empty.
fn take_window(site: usize) -> Taken {
    LOCAL.with(|local| {
        let mut local = local.borrow_mut();
        let taken = Taken {
            ns: local.ns[site],
            census: local.census[site],
            total_ns: local.total_ns[site],
            n: local.window[site],
            refused_n: local.refused[site],
        };
        local.ns[site] = [0; REGION_COUNT];
        local.census[site] = [0; CENSUS_COUNT];
        local.total_ns[site] = 0;
        local.calls[site] = 0;
        local.window[site] = 0;
        local.refused[site] = 0;
        taken
    })
}

/// One window's line, as the round's boot log carries it.
fn format_line(site: usize, taken: &Taken) -> String {
    let ns = &taken.ns;
    let census = &taken.census;
    let named_ns: u64 = ns.iter().sum();
    let mut fields = String::with_capacity(600);
    for (region, name) in REGION_NAMES.iter().enumerate() {
        fields.push_str(&format!(" {name}_us={:.3}", micros(ns[region])));
    }
    for (slot, name) in CENSUS_NAMES.iter().enumerate() {
        fields.push_str(&format!(" {name}={}", census[slot]));
    }
    format!(
        "ADMIT route={} n={} refused_n={} total_us={:.3} named_us={:.3}{fields}",
        SITE_NAMES[site],
        taken.n,
        taken.refused_n,
        micros(taken.total_ns),
        micros(named_ns),
    )
}

/// Print one window of one route, and reset it.
fn emit_line(site: usize) {
    let taken = take_window(site);
    // The line is built before it is written so a test can read the exact text
    // a round's log will carry without a redirection of the process's stderr.
    eprintln!("{}", format_line(site, &taken));
}

fn micros(ns: u64) -> f64 {
    ns as f64 / 1_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The route table and the region table are the printed vocabulary: a slot
    /// added without a name would print as a field the reader cannot map back.
    #[test]
    fn the_printed_tables_name_every_slot() {
        assert_eq!(SITE_NAMES.len(), SITE_COUNT);
        assert_eq!(REGION_NAMES.len(), REGION_COUNT);
        assert_eq!(CENSUS_NAMES.len(), CENSUS_COUNT);
        assert_eq!(Region::TraceValidate as usize, 0);
        assert_eq!(Region::Resources as usize + 1, REGION_COUNT);
    }

    /// Off is the default: the switch reads the environment once, and every
    /// entry point answers `None` without a clock read when it is off.
    #[test]
    fn the_profile_is_off_unless_asked() {
        assert!(!parse_enabled(None));
        assert!(!parse_enabled(Some("")));
        assert!(!parse_enabled(Some("0")));
        assert!(!parse_enabled(Some("off")));
        assert!(parse_enabled(Some("1")));
        assert!(parse_enabled(Some("on")));
        assert!(parse_enabled(Some("true")));
        assert!(parse_enabled(Some("yes")));
    }

    /// The window and the bars bank into one table: a bar that never opens
    /// leaves its slot zero, and the region count is the sum's own size.
    #[test]
    fn the_region_table_partitions_one_walk() {
        let mut local = Local::new();
        local.ns[SITE_VALIDATE][Region::TraceValidate as usize] = 1_000;
        local.ns[SITE_VALIDATE][Region::Resources as usize] = 2_000;
        let named: u64 = local.ns[SITE_VALIDATE].iter().sum();
        assert_eq!(named, 3_000);
        assert_eq!(micros(named), 3.0);
    }

    /// The line a round reads carries every region, every census field and the
    /// route's own name: a field the printer forgot would look like a region
    /// that cost nothing.
    #[test]
    fn the_line_names_every_region_and_every_census_field() {
        let taken = Taken {
            ns: [0; REGION_COUNT],
            census: [0; CENSUS_COUNT],
            total_ns: 1_000,
            n: 4,
            refused_n: 1,
        };
        let line = format_line(SITE_SUBMIT, &taken);
        assert!(line.starts_with("ADMIT route=submit n=4 refused_n=1 total_us=1.000"));
        for name in REGION_NAMES {
            assert!(line.contains(&format!(" {name}_us=")), "missing {name}");
        }
        for name in CENSUS_NAMES {
            assert!(line.contains(&format!(" {name}=")), "missing {name}");
        }
        let named: u64 = taken.ns.iter().sum();
        assert!(line.contains(&format!(" named_us={:.3}", micros(named))));
    }

    /// A window is *taken*, not read: the counters a reader sums must not be
    /// charged twice, and the next window starts empty.
    #[test]
    fn taking_a_window_leaves_the_table_empty() {
        LOCAL.with(|local| {
            let mut local = local.borrow_mut();
            local.window[SITE_DIRECT] = 2;
            local.total_ns[SITE_DIRECT] = 500;
            local.ns[SITE_DIRECT][Region::Present as usize] = 100;
            local.refused[SITE_DIRECT] = 1;
        });
        let taken = take_window(SITE_DIRECT);
        assert_eq!(taken.n, 2);
        assert_eq!(taken.total_ns, 500);
        assert_eq!(taken.refused_n, 1);
        assert_eq!(taken.ns[Region::Present as usize], 100);
        let again = take_window(SITE_DIRECT);
        assert_eq!(again.n, 0);
        assert_eq!(again.total_ns, 0);
        assert_eq!(again.ns[Region::Present as usize], 0);
    }
}
