use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Weak};

use fearless_simd::Level as SimdLevel;

use crate::decomposition::{Level, WavedecPlan, resolve_levels};
use crate::num::{forward_interior_simd, inverse_linear_simd, inverse_periodized_simd};
use crate::simd::{AnalysisInterior, LinearSynthesis, PeriodizedInterior};
use crate::{Boundary, Wavelet, WaveletError, WaveletNum};

/// A reusable, fixed-length one-level DWT/IDWT plan.
///
/// Buffer-size mistakes are programming errors and cause the `_into` methods
/// to panic. Use the plan's sizing methods to prepare buffers once.
///
/// Plans are returned behind [`Arc`] by [`DwtPlanner::plan_dwt`], so cloning a
/// plan handle is cheap and the same immutable plan can be shared between
/// threads. Each concurrent execution must use distinct output and scratch
/// buffers.
pub trait Dwt<T: WaveletNum>: Send + Sync {
    /// Returns the input and reconstructed signal length fixed by this plan.
    fn signal_len(&self) -> usize;

    /// Returns the required length of each output coefficient band.
    fn coeff_len(&self) -> usize;

    /// Returns the minimum scratch-buffer length.
    fn scratch_len(&self) -> usize;

    /// Allocates and computes `(approximation, detail)` coefficients.
    ///
    /// # Panics
    ///
    /// Panics when `signal.len()` differs from [`Self::signal_len`].
    fn forward(&self, signal: &[T]) -> (Vec<T>, Vec<T>) {
        let mut approx = vec![T::zero(); self.coeff_len()];
        let mut detail = vec![T::zero(); self.coeff_len()];
        let mut scratch = vec![T::zero(); self.scratch_len()];
        self.forward_into(signal, &mut approx, &mut detail, &mut scratch);
        (approx, detail)
    }

    /// Allocates and reconstructs a signal of [`Self::signal_len`] samples.
    ///
    /// # Panics
    ///
    /// Panics unless both coefficient bands have exactly [`Self::coeff_len`]
    /// samples.
    fn inverse(&self, approx: &[T], detail: &[T]) -> Vec<T> {
        let mut out = vec![T::zero(); self.signal_len()];
        let mut scratch = vec![T::zero(); self.scratch_len()];
        self.inverse_into(approx, detail, &mut out, &mut scratch);
        out
    }

    /// Computes one decomposition level without allocating.
    ///
    /// `signal`, `approx`, and `detail` must have exactly the lengths reported
    /// by [`Self::signal_len`] and [`Self::coeff_len`]. `scratch` may be longer
    /// than [`Self::scratch_len`] but not shorter.
    ///
    /// # Panics
    ///
    /// Panics when any buffer violates those length requirements.
    fn forward_into(&self, signal: &[T], approx: &mut [T], detail: &mut [T], scratch: &mut [T]);

    /// Reconstructs the plan's original signal length without allocating.
    ///
    /// `approx`, `detail`, and `out` must have exactly the lengths reported by
    /// [`Self::coeff_len`] and [`Self::signal_len`]. `scratch` may be longer
    /// than [`Self::scratch_len`] but not shorter.
    ///
    /// # Panics
    ///
    /// Panics when any buffer violates those length requirements.
    fn inverse_into(&self, approx: &[T], detail: &[T], out: &mut [T], scratch: &mut [T]);
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PlanKey {
    signal_len: usize,
    wavelet_id: u64,
    boundary: Boundary,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct MultilevelPlanKey {
    signal_len: usize,
    wavelet_id: u64,
    boundary: Boundary,
    levels: usize,
}

/// Creates and caches fixed-length discrete wavelet transform plans.
///
/// The planner detects the best available safe SIMD backend once. Repeated
/// requests using the same live [`Wavelet`] and transform configuration share
/// the cached plan.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use wavelets::{Boundary, DwtPlanner, Wavelet};
///
/// let wavelet = Wavelet::haar();
/// let mut planner = DwtPlanner::<f64>::new();
/// let first = planner.plan_dwt(128, &wavelet, Boundary::Periodization)?;
/// let second = planner.plan_dwt(128, &wavelet, Boundary::Periodization)?;
/// assert!(Arc::ptr_eq(&first, &second));
/// # Ok::<(), wavelets::WaveletError>(())
/// ```
pub struct DwtPlanner<T: WaveletNum> {
    cache: HashMap<PlanKey, Weak<dyn Dwt<T>>>,
    multilevel_cache: HashMap<MultilevelPlanKey, Weak<WavedecPlan<T>>>,
    simd_level: SimdLevel,
    marker: PhantomData<T>,
}

impl<T: WaveletNum> DwtPlanner<T> {
    /// Constructs an empty planner.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            multilevel_cache: HashMap::new(),
            simd_level: SimdLevel::new(),
            marker: PhantomData,
        }
    }

    /// Plans a one-level transform for signals of exactly `len` samples.
    ///
    /// Planning validates the boundary/length combination and prepares the
    /// edge-extension and polyphase filter layouts. Repeated identical requests
    /// reuse the same live plan.
    ///
    /// # Errors
    ///
    /// Returns [`WaveletError::EmptySignal`] for `len == 0`, or
    /// [`WaveletError::BoundaryRequiresLongerSignal`] when the selected
    /// extension mode is undefined for `len`.
    pub fn plan_dwt(
        &mut self,
        len: usize,
        wavelet: &Wavelet,
        boundary: Boundary,
    ) -> Result<Arc<dyn Dwt<T>>, WaveletError> {
        let key = PlanKey {
            signal_len: len,
            wavelet_id: wavelet.id(),
            boundary,
        };
        if let Some(plan) = self.cache.get(&key).and_then(Weak::upgrade) {
            return Ok(plan);
        }

        let plan: Arc<dyn Dwt<T>> =
            Arc::new(create_dwt_plan(len, wavelet, boundary, self.simd_level)?);
        self.cache.insert(key, Arc::downgrade(&plan));
        Ok(plan)
    }

    /// Plans a multilevel transform for signals of exactly `len` samples.
    ///
    /// Every single-level plan, band offset, and scratch region is prepared up
    /// front. Repeated requests resolving to the same number of levels reuse
    /// the same live plan.
    ///
    /// # Errors
    ///
    /// Returns [`WaveletError::EmptySignal`] for `len == 0`,
    /// [`WaveletError::InvalidLevel`] when an exact level exceeds the maximum,
    /// or a boundary/length planning error at an intermediate level.
    pub fn plan_wavedec(
        &mut self,
        len: usize,
        wavelet: &Wavelet,
        boundary: Boundary,
        level: Level,
    ) -> Result<Arc<WavedecPlan<T>>, WaveletError> {
        let levels = resolve_levels(len, wavelet.filter_len(), level)?;
        let key = MultilevelPlanKey {
            signal_len: len,
            wavelet_id: wavelet.id(),
            boundary,
            levels,
        };
        if let Some(plan) = self.multilevel_cache.get(&key).and_then(Weak::upgrade) {
            return Ok(plan);
        }

        let filters = PreparedFilterBank::new(wavelet, boundary == Boundary::Periodization);
        let plan = Arc::new(WavedecPlan::new(
            len,
            wavelet,
            boundary,
            levels,
            filters,
            self.simd_level,
        )?);
        self.multilevel_cache.insert(key, Arc::downgrade(&plan));
        Ok(plan)
    }
}

impl<T: WaveletNum> Default for DwtPlanner<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn validate_plan(len: usize, boundary: Boundary) -> Result<(), WaveletError> {
    if len == 0 {
        return Err(WaveletError::EmptySignal);
    }
    if len == 1 && matches!(boundary, Boundary::Reflect | Boundary::Antireflect) {
        return Err(WaveletError::BoundaryRequiresLongerSignal {
            len,
            minimum: 2,
            boundary: boundary.as_str(),
        });
    }
    Ok(())
}

pub(crate) fn create_dwt_plan<T: WaveletNum>(
    len: usize,
    wavelet: &Wavelet,
    boundary: Boundary,
    simd_level: SimdLevel,
) -> Result<PlannedDwt<T>, WaveletError> {
    validate_plan(len, boundary)?;
    let filters = PreparedFilterBank::new(wavelet, boundary == Boundary::Periodization);
    Ok(PlannedDwt::new(len, boundary, filters, simd_level))
}

#[derive(Debug)]
struct InteriorAnalysis {
    first_newest: usize,
    output_len: usize,
}

#[derive(Debug)]
struct AnalysisPlan<T> {
    edges: EdgePlan<T>,
    prefix_len: usize,
    interior: Option<InteriorAnalysis>,
}

#[derive(Debug)]
struct EdgePlan<T> {
    // Each row is the filter-composed finite-boundary transform for one
    // approximation/detail coefficient pair. Both channels share every input
    // load, and repeated references to the same finite sample are coalesced.
    row_offsets: Box<[usize]>,
    terms: Box<[EdgeTerm<T>]>,
}

#[derive(Clone, Copy, Debug)]
struct EdgeTerm<T> {
    input: usize,
    low: T,
    high: T,
}

#[derive(Clone, Copy, Debug)]
struct PeriodizedSynthesis {
    initial_coefficient: usize,
    simd_start: usize,
    simd_available: usize,
    phases_are_swapped: bool,
}

impl PeriodizedSynthesis {
    fn new(signal_len: usize, coeff_len: usize, filter_len: usize) -> Self {
        let half_filter_len = filter_len / 2;
        let shift = half_filter_len - 1;
        let phases_are_swapped = periodized_phases_are_swapped(filter_len);
        let initial_coefficient = (shift / 2) % coeff_len;
        let complete_pairs = signal_len / 2;
        let (simd_start, simd_available) = if coeff_len >= half_filter_len {
            let start = half_filter_len - 1 - initial_coefficient;
            let final_coefficient = coeff_len - usize::from(phases_are_swapped);
            let available = final_coefficient
                .saturating_sub(half_filter_len - 1)
                .min(complete_pairs.saturating_sub(start));
            (start, available)
        } else {
            (0, 0)
        };
        Self {
            initial_coefficient,
            simd_start,
            simd_available,
            phases_are_swapped,
        }
    }
}

fn periodized_phases_are_swapped(filter_len: usize) -> bool {
    !(filter_len / 2 - 1).is_multiple_of(2)
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedFilterBank<T> {
    data: Arc<[T]>,
    filter_len: usize,
}

impl<T: WaveletNum> PreparedFilterBank<T> {
    pub(crate) fn new(wavelet: &Wavelet, periodized: bool) -> Self {
        let filter_len = wavelet.filter_len();
        let mut data = Vec::with_capacity(4 * filter_len);
        data.extend(wavelet.dec_lo().iter().copied().map(T::from_f64));
        data.extend(wavelet.dec_hi().iter().copied().map(T::from_f64));

        let rec_lo_start = data.len();
        extend_polyphase(&mut data, wavelet.rec_lo());
        if periodized && periodized_phases_are_swapped(filter_len) {
            data[rec_lo_start..].rotate_left(filter_len / 2);
        }

        let rec_hi_start = data.len();
        extend_polyphase(&mut data, wavelet.rec_hi());
        if periodized && periodized_phases_are_swapped(filter_len) {
            data[rec_hi_start..].rotate_left(filter_len / 2);
        }

        Self {
            data: data.into(),
            filter_len,
        }
    }

    fn analysis(&self) -> (&[T], &[T]) {
        let (dec_lo, remaining) = self.data.split_at(self.filter_len);
        let (dec_hi, _) = remaining.split_at(self.filter_len);
        (dec_lo, dec_hi)
    }

    fn synthesis(&self) -> (&[T], &[T]) {
        let synthesis = &self.data[2 * self.filter_len..];
        synthesis.split_at(self.filter_len)
    }
}

#[derive(Debug)]
pub(crate) struct PlannedDwt<T> {
    signal_len: usize,
    coeff_len: usize,
    filters: PreparedFilterBank<T>,
    analysis: AnalysisPlan<T>,
    periodized_synthesis: Option<PeriodizedSynthesis>,
    simd_level: SimdLevel,
}

impl<T: WaveletNum> PlannedDwt<T> {
    pub(crate) fn new(
        signal_len: usize,
        boundary: Boundary,
        filters: PreparedFilterBank<T>,
        simd_level: SimdLevel,
    ) -> Self {
        let filter_len = filters.filter_len;
        let coeff_len = coefficient_len(signal_len, filter_len, boundary);
        let periodized_synthesis = (boundary == Boundary::Periodization)
            .then(|| PeriodizedSynthesis::new(signal_len, coeff_len, filter_len));
        let (dec_lo, dec_hi) = filters.analysis();
        let analysis = build_analysis(signal_len, coeff_len, dec_lo, dec_hi, boundary);
        Self {
            signal_len,
            coeff_len,
            filters,
            analysis,
            periodized_synthesis,
            simd_level,
        }
    }

    fn inverse_linear(&self, approx: &[T], detail: &[T], out: &mut [T]) {
        let (rec_lo, rec_hi) = self.filters.synthesis();
        let half_filter_len = rec_lo.len() / 2;
        let (even_lo, odd_lo) = rec_lo.split_at(half_filter_len);
        let (even_hi, odd_hi) = rec_hi.split_at(half_filter_len);

        let vectorized_pairs = inverse_linear_simd(
            self.simd_level,
            LinearSynthesis {
                rec_lo,
                rec_hi,
                approx,
                detail,
            },
            out,
        );

        // Cropping the full convolution by `filter_len - 2` makes each output
        // pair consume the same reversed coefficient window. Fusing both
        // polyphase dots keeps that window hot and loads it only once.
        for (tail_coefficient, samples) in out[2 * vectorized_pairs..].chunks_mut(2).enumerate() {
            let coefficient = vectorized_pairs + tail_coefficient;
            let coefficient_end = coefficient + half_filter_len;
            let approx = &approx[coefficient..coefficient_end];
            let detail = &detail[coefficient..coefficient_end];
            let (first, second) = synthesis_pair(even_lo, even_hi, odd_lo, odd_hi, approx, detail);
            samples[0] = first;
            if samples.len() == 2 {
                samples[1] = second;
            }
        }
    }

    fn inverse_periodized(
        &self,
        layout: PeriodizedSynthesis,
        approx: &[T],
        detail: &[T],
        out: &mut [T],
    ) {
        let (rec_lo, rec_hi) = self.filters.synthesis();
        let half_filter_len = rec_lo.len() / 2;
        let (first_lo, second_lo) = rec_lo.split_at(half_filter_len);
        let (first_hi, second_hi) = rec_hi.split_at(half_filter_len);

        let (scalar_prefix, remainder) = out[..2 * layout.simd_start].as_chunks_mut::<2>();
        debug_assert!(remainder.is_empty());
        for (pair, samples) in scalar_prefix.iter_mut().enumerate() {
            let first_coefficient = (layout.initial_coefficient + pair) % self.coeff_len;
            let (first, second) = synthesis_periodized_pair(
                (first_lo, first_hi),
                (second_lo, second_hi),
                approx,
                detail,
                first_coefficient,
                layout.phases_are_swapped,
            );
            samples[0] = first;
            samples[1] = second;
        }

        let vectorized = inverse_periodized_simd(
            self.simd_level,
            PeriodizedInterior {
                first_lo,
                first_hi,
                second_lo,
                second_hi,
                approx,
                detail,
                first_coefficient: half_filter_len - 1,
                second_offset: usize::from(layout.phases_are_swapped),
            },
            &mut out[2 * layout.simd_start..2 * (layout.simd_start + layout.simd_available)],
        );

        let scalar_start = layout.simd_start + vectorized;
        for (tail_pair, samples) in out[2 * scalar_start..].chunks_mut(2).enumerate() {
            let pair = scalar_start + tail_pair;
            let first_coefficient = (layout.initial_coefficient + pair) % self.coeff_len;
            let (first, second) = synthesis_periodized_pair(
                (first_lo, first_hi),
                (second_lo, second_hi),
                approx,
                detail,
                first_coefficient,
                layout.phases_are_swapped,
            );
            samples[0] = first;
            if samples.len() == 2 {
                samples[1] = second;
            }
        }
    }
}

impl<T: WaveletNum> Dwt<T> for PlannedDwt<T> {
    fn signal_len(&self) -> usize {
        self.signal_len
    }

    fn coeff_len(&self) -> usize {
        self.coeff_len
    }

    fn scratch_len(&self) -> usize {
        0
    }

    fn forward_into(&self, signal: &[T], approx: &mut [T], detail: &mut [T], scratch: &mut [T]) {
        assert_eq!(signal.len(), self.signal_len, "incorrect signal length");
        assert_eq!(
            approx.len(),
            self.coeff_len,
            "incorrect approximation length"
        );
        assert_eq!(detail.len(), self.coeff_len, "incorrect detail length");
        assert!(
            scratch.len() >= self.scratch_len(),
            "scratch buffer is too small"
        );

        let (dec_lo, dec_hi) = self.filters.analysis();
        let prefix_len = self.analysis.prefix_len;
        analyze_edges(
            signal,
            &self.analysis.edges,
            0,
            &mut approx[..prefix_len],
            &mut detail[..prefix_len],
        );

        let mut suffix_start = prefix_len;
        if let Some(interior) = &self.analysis.interior {
            let interior_end = prefix_len + interior.output_len;
            let interior_approx = &mut approx[prefix_len..interior_end];
            let interior_detail = &mut detail[prefix_len..interior_end];
            let vectorized = forward_interior_simd(
                self.simd_level,
                AnalysisInterior {
                    dec_lo,
                    dec_hi,
                    signal,
                    first_newest: interior.first_newest,
                },
                interior_approx,
                interior_detail,
            );

            for output in vectorized..interior.output_len {
                let newest = interior.first_newest + 2 * output;
                let (low, high) = analyze_interior(signal, newest, dec_lo, dec_hi);
                interior_approx[output] = low;
                interior_detail[output] = high;
            }
            suffix_start = interior_end;
        }

        analyze_edges(
            signal,
            &self.analysis.edges,
            prefix_len,
            &mut approx[suffix_start..],
            &mut detail[suffix_start..],
        );
    }

    fn inverse_into(&self, approx: &[T], detail: &[T], out: &mut [T], scratch: &mut [T]) {
        assert_eq!(
            approx.len(),
            self.coeff_len,
            "incorrect approximation length"
        );
        assert_eq!(detail.len(), self.coeff_len, "incorrect detail length");
        assert_eq!(out.len(), self.signal_len, "incorrect output length");
        assert!(
            scratch.len() >= self.scratch_len(),
            "scratch buffer is too small"
        );

        if let Some(layout) = self.periodized_synthesis {
            self.inverse_periodized(layout, approx, detail, out);
        } else {
            self.inverse_linear(approx, detail, out);
        }
    }
}

fn extend_polyphase<T: WaveletNum>(out: &mut Vec<T>, filter: &[f64]) {
    debug_assert!(filter.len().is_multiple_of(2));
    out.extend(
        filter
            .iter()
            .step_by(2)
            .chain(filter.iter().skip(1).step_by(2))
            .copied()
            .map(T::from_f64),
    );
}

#[inline(always)]
fn synthesis_pair<T: WaveletNum>(
    even_lo: &[T],
    even_hi: &[T],
    odd_lo: &[T],
    odd_hi: &[T],
    approx: &[T],
    detail: &[T],
) -> (T, T) {
    let mut even = T::zero();
    let mut odd = T::zero();
    for ((((even_low, even_high), (odd_low, odd_high)), approximation), detail) in even_lo
        .iter()
        .zip(even_hi)
        .zip(odd_lo.iter().zip(odd_hi))
        .zip(approx.iter().rev())
        .zip(detail.iter().rev())
    {
        even += *even_low * *approximation;
        even += *even_high * *detail;
        odd += *odd_low * *approximation;
        odd += *odd_high * *detail;
    }
    (even, odd)
}

#[inline(always)]
fn synthesis_pair_windows<T: WaveletNum>(
    (first_lo, first_hi): (&[T], &[T]),
    (second_lo, second_hi): (&[T], &[T]),
    (first_approx, first_detail): (&[T], &[T]),
    (second_approx, second_detail): (&[T], &[T]),
) -> (T, T) {
    let mut first = T::zero();
    let mut second = T::zero();
    for (
        ((first_low, first_high), (second_low, second_high)),
        ((first_approx, first_detail), (second_approx, second_detail)),
    ) in first_lo
        .iter()
        .zip(first_hi)
        .zip(second_lo.iter().zip(second_hi))
        .zip(
            first_approx
                .iter()
                .rev()
                .zip(first_detail.iter().rev())
                .zip(second_approx.iter().rev().zip(second_detail.iter().rev())),
        )
    {
        first += *first_low * *first_approx;
        first += *first_high * *first_detail;
        second += *second_low * *second_approx;
        second += *second_high * *second_detail;
    }
    (first, second)
}

#[inline(always)]
fn synthesis_pair_cyclic<T: WaveletNum>(
    (first_lo, first_hi): (&[T], &[T]),
    (second_lo, second_hi): (&[T], &[T]),
    approx: &[T],
    detail: &[T],
    first_coefficient: usize,
    second_coefficient: usize,
) -> (T, T) {
    let mut first = T::zero();
    let mut second = T::zero();
    let mut first_coefficient = first_coefficient;
    let mut second_coefficient = second_coefficient;
    for (((first_low, first_high), second_low), second_high) in
        first_lo.iter().zip(first_hi).zip(second_lo).zip(second_hi)
    {
        first += *first_low * approx[first_coefficient];
        first += *first_high * detail[first_coefficient];
        second += *second_low * approx[second_coefficient];
        second += *second_high * detail[second_coefficient];
        first_coefficient = decrement_wrapping(first_coefficient, approx.len());
        second_coefficient = decrement_wrapping(second_coefficient, approx.len());
    }
    (first, second)
}

#[inline(always)]
fn synthesis_periodized_pair<T: WaveletNum>(
    first_filters: (&[T], &[T]),
    second_filters: (&[T], &[T]),
    approx: &[T],
    detail: &[T],
    first_coefficient: usize,
    phases_are_swapped: bool,
) -> (T, T) {
    let second_coefficient = if phases_are_swapped {
        increment_wrapping(first_coefficient, approx.len())
    } else {
        first_coefficient
    };
    let filter_len = first_filters.0.len();

    if first_coefficient + 1 >= filter_len && second_coefficient + 1 >= filter_len {
        let first_start = first_coefficient + 1 - filter_len;
        let second_start = second_coefficient + 1 - filter_len;
        synthesis_pair_windows(
            first_filters,
            second_filters,
            (
                &approx[first_start..=first_coefficient],
                &detail[first_start..=first_coefficient],
            ),
            (
                &approx[second_start..=second_coefficient],
                &detail[second_start..=second_coefficient],
            ),
        )
    } else {
        synthesis_pair_cyclic(
            first_filters,
            second_filters,
            approx,
            detail,
            first_coefficient,
            second_coefficient,
        )
    }
}

#[inline]
fn increment_wrapping(value: usize, len: usize) -> usize {
    if value + 1 == len { 0 } else { value + 1 }
}

#[inline]
fn decrement_wrapping(value: usize, len: usize) -> usize {
    if value == 0 { len - 1 } else { value - 1 }
}

pub(crate) fn coefficient_len(signal_len: usize, filter_len: usize, boundary: Boundary) -> usize {
    if boundary == Boundary::Periodization {
        signal_len.div_ceil(2)
    } else {
        (signal_len + filter_len - 1) / 2
    }
}

fn build_analysis<T: WaveletNum>(
    signal_len: usize,
    coeff_len: usize,
    dec_lo: &[T],
    dec_hi: &[T],
    boundary: Boundary,
) -> AnalysisPlan<T> {
    debug_assert_eq!(dec_lo.len(), dec_hi.len());
    let filter_len = dec_lo.len();
    let phase = if boundary == Boundary::Periodization {
        filter_len / 2
    } else {
        1
    };
    let is_interior = |coefficient: usize| {
        let newest = (2 * coefficient + phase) as isize;
        let oldest = newest - (filter_len - 1) as isize;
        oldest >= 0 && newest < signal_len as isize
    };
    let interior_start = (0..coeff_len).find(|&coefficient| is_interior(coefficient));
    let interior_end = interior_start.map_or(coeff_len, |start| {
        start
            + (start..coeff_len)
                .take_while(|&coefficient| is_interior(coefficient))
                .count()
    });
    debug_assert!((interior_end..coeff_len).all(|coefficient| !is_interior(coefficient)));

    let prefix_end = interior_start.unwrap_or(coeff_len);
    let edge_count = prefix_end + coeff_len - interior_end;
    let mut row_offsets = Vec::with_capacity(edge_count + 1);
    let mut terms = Vec::<EdgeTerm<T>>::with_capacity(edge_count * filter_len);
    // A dense position map avoids hashing when it is no larger than the raw
    // edge-rule grid. Large signals retain an O(filter_len) sparse planner.
    let dense_position_limit = edge_count.saturating_mul(filter_len);
    let mut dense_positions =
        (signal_len <= dense_position_limit).then(|| vec![usize::MAX; signal_len]);
    let mut sparse_positions = dense_positions
        .is_none()
        .then(|| HashMap::<usize, usize>::with_capacity(3 * filter_len));
    row_offsets.push(0);
    for coefficient in (0..prefix_end).chain(interior_end..coeff_len) {
        if let Some(positions) = &mut sparse_positions {
            positions.clear();
        }
        let row_start = terms.len();
        let newest = (2 * coefficient + phase) as isize;
        for tap in 0..filter_len {
            for_each_extension_term(
                newest - tap as isize,
                signal_len,
                boundary,
                |input, weight| {
                    let low = dec_lo[tap] * weight;
                    let high = dec_hi[tap] * weight;
                    let position = if let Some(positions) = &mut dense_positions {
                        let position = positions[input];
                        if position != usize::MAX && position >= row_start {
                            Some(position)
                        } else {
                            positions[input] = terms.len();
                            None
                        }
                    } else {
                        let positions = sparse_positions
                            .as_mut()
                            .expect("one edge position map is always available");
                        if let Some(&position) = positions.get(&input) {
                            Some(position)
                        } else {
                            positions.insert(input, terms.len());
                            None
                        }
                    };
                    if let Some(position) = position {
                        terms[position].low += low;
                        terms[position].high += high;
                    } else {
                        terms.push(EdgeTerm { input, low, high });
                    }
                },
            );
        }
        row_offsets.push(terms.len());
    }

    AnalysisPlan {
        edges: EdgePlan {
            row_offsets: row_offsets.into_boxed_slice(),
            terms: terms.into_boxed_slice(),
        },
        prefix_len: prefix_end,
        interior: interior_start.map(|start| InteriorAnalysis {
            first_newest: 2 * start + phase,
            output_len: interior_end - start,
        }),
    }
}

#[inline]
fn analyze_interior<T: WaveletNum>(
    signal: &[T],
    newest: usize,
    dec_lo: &[T],
    dec_hi: &[T],
) -> (T, T) {
    let mut low = T::zero();
    let mut high = T::zero();
    for tap in 0..dec_lo.len() {
        let sample = signal[newest - tap];
        low += dec_lo[tap] * sample;
        high += dec_hi[tap] * sample;
    }
    (low, high)
}

#[inline]
fn analyze_edges<T: WaveletNum>(
    signal: &[T],
    edges: &EdgePlan<T>,
    first_row: usize,
    approx: &mut [T],
    detail: &mut [T],
) {
    debug_assert_eq!(approx.len(), detail.len());
    debug_assert!(first_row + approx.len() < edges.row_offsets.len());
    for (row, (approximation, detail)) in
        (first_row..).zip(approx.iter_mut().zip(detail.iter_mut()))
    {
        let mut low = T::zero();
        let mut high = T::zero();
        for term in &edges.terms[edges.row_offsets[row]..edges.row_offsets[row + 1]] {
            let sample = signal[term.input];
            low += term.low * sample;
            high += term.high * sample;
        }
        *approximation = low;
        *detail = high;
    }
}

fn for_each_extension_term<T: WaveletNum>(
    index: isize,
    signal_len: usize,
    boundary: Boundary,
    mut visit: impl FnMut(usize, T),
) {
    // Every supported extension is a linear map from the finite signal to one
    // requested sample. Planning composes these terms with both analysis
    // filters; execution never needs to know which boundary mode produced it.
    let weight = |value: f64| T::from_f64(value);
    if (0..signal_len as isize).contains(&index) {
        visit(index as usize, weight(1.0));
        return;
    }

    match boundary {
        Boundary::Zero => {}
        Boundary::Constant => {
            visit(if index < 0 { 0 } else { signal_len - 1 }, weight(1.0));
        }
        Boundary::Periodic => {
            visit(index.rem_euclid(signal_len as isize) as usize, weight(1.0));
        }
        Boundary::Periodization => {
            let periodic_len = signal_len + signal_len % 2;
            let wrapped = index.rem_euclid(periodic_len as isize) as usize;
            visit(wrapped.min(signal_len - 1), weight(1.0));
        }
        Boundary::Symmetric => {
            let period = 2 * signal_len;
            let wrapped = index.rem_euclid(period as isize) as usize;
            let reflected = if wrapped < signal_len {
                wrapped
            } else {
                period - 1 - wrapped
            };
            visit(reflected, weight(1.0));
        }
        Boundary::Antisymmetric => {
            let period = 2 * signal_len;
            let wrapped = index.rem_euclid(period as isize) as usize;
            if wrapped < signal_len {
                visit(wrapped, weight(1.0));
            } else {
                visit(period - 1 - wrapped, weight(-1.0));
            }
        }
        Boundary::Reflect => {
            let span = signal_len - 1;
            let period = 2 * span;
            let wrapped = index.rem_euclid(period as isize) as usize;
            let reflected = if wrapped < signal_len {
                wrapped
            } else {
                period - wrapped
            };
            visit(reflected, weight(1.0));
        }
        Boundary::Smooth => {
            if signal_len == 1 {
                visit(0, weight(1.0));
            } else if index < 0 {
                let distance = (-index) as f64;
                visit(0, weight(1.0 + distance));
                visit(1, weight(-distance));
            } else {
                let distance = (index - (signal_len - 1) as isize) as f64;
                visit(signal_len - 1, weight(1.0 + distance));
                visit(signal_len - 2, weight(-distance));
            }
        }
        Boundary::Antireflect => for_each_antireflect_term(index, signal_len, visit),
    }
}

fn for_each_antireflect_term<T: WaveletNum>(
    index: isize,
    signal_len: usize,
    mut visit: impl FnMut(usize, T),
) {
    debug_assert!(signal_len >= 2);
    debug_assert!(!(0..signal_len as isize).contains(&index));

    let last = signal_len - 1;
    let distance = if index < 0 {
        (-index) as usize
    } else {
        index as usize - last
    };
    let segment = (distance - 1) / last;
    let offset = (distance - 1) % last + 1;
    let weight = |value: isize| T::from_f64(value as f64);

    if index < 0 {
        if segment == 0 {
            visit(0, weight(2));
            visit(offset, weight(-1));
        } else if segment.is_multiple_of(2) {
            visit(0, weight(segment as isize + 2));
            visit(last, weight(-(segment as isize)));
            visit(offset, weight(-1));
        } else {
            visit(0, weight(segment as isize + 1));
            visit(last, weight(-(segment as isize) - 1));
            visit(last - offset, weight(1));
        }
    } else if segment == 0 {
        visit(last, weight(2));
        visit(last - offset, weight(-1));
    } else if segment.is_multiple_of(2) {
        visit(0, weight(-(segment as isize)));
        visit(last, weight(segment as isize + 2));
        visit(last - offset, weight(-1));
    } else {
        visit(0, weight(-(segment as isize) - 1));
        visit(last, weight(segment as isize + 1));
        visit(offset, weight(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_reuses_live_plans() {
        let mut planner = DwtPlanner::<f64>::new();
        let wavelet = Wavelet::haar();
        let first = planner.plan_dwt(8, &wavelet, Boundary::Symmetric).unwrap();
        let second = planner.plan_dwt(8, &wavelet, Boundary::Symmetric).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn planner_reuses_equivalent_live_multilevel_plans() {
        let mut planner = DwtPlanner::<f64>::new();
        let wavelet = Wavelet::haar();
        let first = planner
            .plan_wavedec(16, &wavelet, Boundary::Symmetric, Level::Max)
            .unwrap();
        let second = planner
            .plan_wavedec(16, &wavelet, Boundary::Symmetric, Level::Exact(4))
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn reflect_rejects_a_single_sample() {
        let mut planner = DwtPlanner::<f64>::new();
        let error = match planner.plan_dwt(1, &Wavelet::haar(), Boundary::Reflect) {
            Ok(_) => panic!("length-one reflect plan unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            WaveletError::BoundaryRequiresLongerSignal { .. }
        ));
    }

    #[test]
    fn antireflect_terms_cover_repeated_reflections() {
        let signal = [1.0_f64, 3.0, 6.0];
        let expected = [
            -29.0, -27.0, -24.0, -21.0, -19.0, -17.0, -14.0, -11.0, -9.0, -7.0, -4.0, -1.0, 1.0,
            3.0, 6.0, 9.0, 11.0, 13.0, 16.0, 19.0, 21.0, 23.0, 26.0, 29.0, 31.0, 33.0, 36.0,
        ];

        let actual: Vec<_> = (-12..=14)
            .map(|index| extended_sample(&signal, index, Boundary::Antireflect))
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn two_sample_antireflect_is_linear_extrapolation() {
        let signal = [-3.0_f64, 5.0];
        for index in -128..=128 {
            let actual = extended_sample(&signal, index, Boundary::Antireflect);
            let expected = signal[0] + index as f64 * (signal[1] - signal[0]);
            assert_eq!(actual, expected, "extended sample {index}");
        }
    }

    #[test]
    fn compiled_edge_rows_coalesce_repeated_inputs() {
        let wavelet = Wavelet::coiflet(17).unwrap();
        let plan =
            create_dwt_plan::<f64>(16, &wavelet, Boundary::Antireflect, SimdLevel::new()).unwrap();

        for offsets in plan.analysis.edges.row_offsets.windows(2) {
            let row = &plan.analysis.edges.terms[offsets[0]..offsets[1]];
            assert!(row.len() <= plan.signal_len);
            for (position, term) in row.iter().enumerate() {
                assert!(
                    row[..position]
                        .iter()
                        .all(|earlier| earlier.input != term.input),
                    "edge row contains input {} more than once",
                    term.input
                );
            }
        }
    }

    fn extended_sample(signal: &[f64], index: isize, boundary: Boundary) -> f64 {
        let mut value = 0.0;
        for_each_extension_term::<f64>(index, signal.len(), boundary, |input, weight| {
            value += signal[input] * weight;
        });
        value
    }
}
