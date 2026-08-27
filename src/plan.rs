use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Weak};

use fearless_simd::Level as SimdLevel;

use crate::decomposition::{Level, WavedecPlan, resolve_levels};
use crate::num::inverse_linear_simd;
use crate::{Boundary, Wavelet, WaveletError, WaveletNum};

/// A reusable, fixed-length one-level DWT/IDWT plan.
///
/// Buffer-size mistakes are programming errors and cause the `_into` methods
/// to panic. Use the plan's sizing methods to prepare buffers once.
pub trait Dwt<T: WaveletNum>: Send + Sync {
    /// Returns the input and reconstructed signal length fixed by this plan.
    fn signal_len(&self) -> usize;

    /// Returns the required length of each output coefficient band.
    fn coeff_len(&self) -> usize;

    /// Returns the minimum scratch-buffer length.
    fn scratch_len(&self) -> usize;

    /// Allocates and computes `(approximation, detail)` coefficients.
    fn forward(&self, signal: &[T]) -> (Vec<T>, Vec<T>) {
        let mut approx = vec![T::zero(); self.coeff_len()];
        let mut detail = vec![T::zero(); self.coeff_len()];
        let mut scratch = vec![T::zero(); self.scratch_len()];
        self.forward_into(signal, &mut approx, &mut detail, &mut scratch);
        (approx, detail)
    }

    /// Allocates and reconstructs a signal of [`Self::signal_len`] samples.
    fn inverse(&self, approx: &[T], detail: &[T]) -> Vec<T> {
        let mut out = vec![T::zero(); self.signal_len()];
        let mut scratch = vec![T::zero(); self.scratch_len()];
        self.inverse_into(approx, detail, &mut out, &mut scratch);
        out
    }

    /// Computes one decomposition level without allocating.
    fn forward_into(&self, signal: &[T], approx: &mut [T], detail: &mut [T], scratch: &mut [T]);

    /// Reconstructs the plan's original signal length without allocating.
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
    pub fn plan_dwt(
        &mut self,
        len: usize,
        wavelet: &Wavelet,
        boundary: Boundary,
    ) -> Result<Arc<dyn Dwt<T>>, WaveletError> {
        validate_plan(len, boundary)?;
        let key = PlanKey {
            signal_len: len,
            wavelet_id: wavelet.id(),
            boundary,
        };
        if let Some(plan) = self.cache.get(&key).and_then(Weak::upgrade) {
            return Ok(plan);
        }

        let plan: Arc<dyn Dwt<T>> =
            Arc::new(PlannedDwt::new(len, wavelet, boundary, self.simd_level));
        self.cache.insert(key, Arc::downgrade(&plan));
        Ok(plan)
    }

    /// Plans a multilevel transform for signals of exactly `len` samples.
    ///
    /// Every single-level plan, band offset, and scratch region is prepared up
    /// front. Repeated requests resolving to the same number of levels reuse
    /// the same live plan.
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

        let plan = Arc::new(WavedecPlan::new(self, len, wavelet, boundary, levels)?);
        self.multilevel_cache.insert(key, Arc::downgrade(&plan));
        Ok(plan)
    }
}

impl<T: WaveletNum> Default for DwtPlanner<T> {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_plan(len: usize, boundary: Boundary) -> Result<(), WaveletError> {
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

#[derive(Debug)]
struct EdgeOutput {
    samples: Box<[SampleRule]>,
}

#[derive(Debug)]
enum AnalysisOutput {
    Interior { newest: usize },
    Edge(EdgeOutput),
}

#[derive(Clone, Copy, Debug)]
enum SampleRule {
    Zero,
    Direct { index: usize, negative: bool },
    SmoothLeft { distance: usize },
    SmoothRight { distance: usize },
    Antireflect { index: isize },
}

#[derive(Debug)]
struct PlannedDwt<T> {
    signal_len: usize,
    coeff_len: usize,
    dec_lo: Box<[T]>,
    dec_hi: Box<[T]>,
    rec_lo: Box<[T]>,
    rec_hi: Box<[T]>,
    analysis: Box<[AnalysisOutput]>,
    periodization: bool,
    simd_level: SimdLevel,
}

impl<T: WaveletNum> PlannedDwt<T> {
    fn new(
        signal_len: usize,
        wavelet: &Wavelet,
        boundary: Boundary,
        simd_level: SimdLevel,
    ) -> Self {
        let filter_len = wavelet.filter_len();
        let coeff_len = coefficient_len(signal_len, filter_len, boundary);
        let dec_lo: Box<[_]> = wavelet.dec_lo().iter().copied().map(T::from_f64).collect();
        let dec_hi: Box<[_]> = wavelet.dec_hi().iter().copied().map(T::from_f64).collect();
        let rec_lo = polyphase_filter(wavelet.rec_lo());
        let rec_hi = polyphase_filter(wavelet.rec_hi());
        let analysis = build_analysis(signal_len, coeff_len, filter_len, boundary);
        Self {
            signal_len,
            coeff_len,
            dec_lo,
            dec_hi,
            rec_lo,
            rec_hi,
            analysis,
            periodization: boundary == Boundary::Periodization,
            simd_level,
        }
    }

    fn inverse_linear(&self, approx: &[T], detail: &[T], out: &mut [T]) {
        let half_filter_len = self.rec_lo.len() / 2;
        let (even_lo, odd_lo) = self.rec_lo.split_at(half_filter_len);
        let (even_hi, odd_hi) = self.rec_hi.split_at(half_filter_len);

        let vectorized_pairs = inverse_linear_simd(
            self.simd_level,
            &self.rec_lo,
            &self.rec_hi,
            approx,
            detail,
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

    fn inverse_periodized(&self, approx: &[T], detail: &[T], out: &mut [T]) {
        let half_filter_len = self.rec_lo.len() / 2;
        let (even_lo, odd_lo) = self.rec_lo.split_at(half_filter_len);
        let (even_hi, odd_hi) = self.rec_hi.split_at(half_filter_len);
        let shift = half_filter_len - 1;
        let phases_are_swapped = !shift.is_multiple_of(2);
        let mut first_coefficient = (shift / 2) % self.coeff_len;
        let (first_lo, first_hi, second_lo, second_hi) = if phases_are_swapped {
            (odd_lo, odd_hi, even_lo, even_hi)
        } else {
            (even_lo, even_hi, odd_lo, odd_hi)
        };

        for samples in out.chunks_mut(2) {
            let second_coefficient = if phases_are_swapped {
                increment_wrapping(first_coefficient, self.coeff_len)
            } else {
                first_coefficient
            };

            // Almost every periodic window is contiguous. Only windows that
            // cross coefficient zero need the cyclic edge kernel.
            let (first, second) = if first_coefficient + 1 >= half_filter_len
                && second_coefficient + 1 >= half_filter_len
            {
                let first_start = first_coefficient + 1 - half_filter_len;
                let second_start = second_coefficient + 1 - half_filter_len;
                synthesis_pair_windows(
                    (first_lo, first_hi),
                    (second_lo, second_hi),
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
                    (first_lo, first_hi),
                    (second_lo, second_hi),
                    approx,
                    detail,
                    first_coefficient,
                    second_coefficient,
                )
            };
            samples[0] = first;
            if samples.len() == 2 {
                samples[1] = second;
            }

            first_coefficient = increment_wrapping(first_coefficient, self.coeff_len);
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

        for (output, (approximation, detail)) in self
            .analysis
            .iter()
            .zip(approx.iter_mut().zip(detail.iter_mut()))
        {
            match output {
                AnalysisOutput::Interior { newest } => {
                    let mut low = T::zero();
                    let mut high = T::zero();
                    for tap in 0..self.dec_lo.len() {
                        let sample = signal[newest - tap];
                        low += self.dec_lo[tap] * sample;
                        high += self.dec_hi[tap] * sample;
                    }
                    *approximation = low;
                    *detail = high;
                }
                AnalysisOutput::Edge(edge) => {
                    let mut low = T::zero();
                    let mut high = T::zero();
                    for (tap, rule) in edge.samples.iter().copied().enumerate() {
                        let sample = evaluate_sample(signal, rule);
                        low += self.dec_lo[tap] * sample;
                        high += self.dec_hi[tap] * sample;
                    }
                    *approximation = low;
                    *detail = high;
                }
            }
        }
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

        if self.periodization {
            self.inverse_periodized(approx, detail, out);
        } else {
            self.inverse_linear(approx, detail, out);
        }
    }
}

fn polyphase_filter<T: WaveletNum>(filter: &[f64]) -> Box<[T]> {
    debug_assert!(filter.len().is_multiple_of(2));
    filter
        .iter()
        .step_by(2)
        .chain(filter.iter().skip(1).step_by(2))
        .copied()
        .map(T::from_f64)
        .collect()
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

fn build_analysis(
    signal_len: usize,
    coeff_len: usize,
    filter_len: usize,
    boundary: Boundary,
) -> Box<[AnalysisOutput]> {
    let phase = if boundary == Boundary::Periodization {
        filter_len / 2
    } else {
        1
    };
    (0..coeff_len)
        .map(|coefficient| {
            let newest = (2 * coefficient + phase) as isize;
            let oldest = newest - (filter_len - 1) as isize;
            if oldest >= 0 && newest < signal_len as isize {
                AnalysisOutput::Interior {
                    newest: newest as usize,
                }
            } else {
                AnalysisOutput::Edge(EdgeOutput {
                    samples: (0..filter_len)
                        .map(|tap| extension_rule(newest - tap as isize, signal_len, boundary))
                        .collect(),
                })
            }
        })
        .collect()
}

fn extension_rule(index: isize, signal_len: usize, boundary: Boundary) -> SampleRule {
    if (0..signal_len as isize).contains(&index) {
        return SampleRule::Direct {
            index: index as usize,
            negative: false,
        };
    }

    match boundary {
        Boundary::Zero => SampleRule::Zero,
        Boundary::Constant => SampleRule::Direct {
            index: if index < 0 { 0 } else { signal_len - 1 },
            negative: false,
        },
        Boundary::Periodic => SampleRule::Direct {
            index: index.rem_euclid(signal_len as isize) as usize,
            negative: false,
        },
        Boundary::Periodization => {
            let periodic_len = signal_len + signal_len % 2;
            let wrapped = index.rem_euclid(periodic_len as isize) as usize;
            SampleRule::Direct {
                index: wrapped.min(signal_len - 1),
                negative: false,
            }
        }
        Boundary::Symmetric => {
            let period = 2 * signal_len;
            let wrapped = index.rem_euclid(period as isize) as usize;
            let reflected = if wrapped < signal_len {
                wrapped
            } else {
                period - 1 - wrapped
            };
            SampleRule::Direct {
                index: reflected,
                negative: false,
            }
        }
        Boundary::Antisymmetric => {
            let period = 2 * signal_len;
            let wrapped = index.rem_euclid(period as isize) as usize;
            if wrapped < signal_len {
                SampleRule::Direct {
                    index: wrapped,
                    negative: false,
                }
            } else {
                SampleRule::Direct {
                    index: period - 1 - wrapped,
                    negative: true,
                }
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
            SampleRule::Direct {
                index: reflected,
                negative: false,
            }
        }
        Boundary::Smooth => {
            if signal_len == 1 {
                SampleRule::Direct {
                    index: 0,
                    negative: false,
                }
            } else if index < 0 {
                SampleRule::SmoothLeft {
                    distance: (-index) as usize,
                }
            } else {
                SampleRule::SmoothRight {
                    distance: (index - (signal_len - 1) as isize) as usize,
                }
            }
        }
        Boundary::Antireflect => SampleRule::Antireflect { index },
    }
}

#[inline]
fn evaluate_sample<T: WaveletNum>(signal: &[T], rule: SampleRule) -> T {
    match rule {
        SampleRule::Zero => T::zero(),
        SampleRule::Direct { index, negative } => {
            if negative {
                T::zero() - signal[index]
            } else {
                signal[index]
            }
        }
        SampleRule::SmoothLeft { distance } => {
            signal[0] + T::from_f64(distance as f64) * (signal[0] - signal[1])
        }
        SampleRule::SmoothRight { distance } => {
            let last = signal.len() - 1;
            signal[last] + T::from_f64(distance as f64) * (signal[last] - signal[last - 1])
        }
        SampleRule::Antireflect { index } => antireflect_sample(signal, index),
    }
}

fn antireflect_sample<T: WaveletNum>(signal: &[T], index: isize) -> T {
    debug_assert!(signal.len() >= 2);
    let target_distance = if index < 0 {
        (-index) as usize
    } else {
        index as usize - (signal.len() - 1)
    };
    let mut distance = 0;
    let mut edge = if index < 0 {
        signal[0]
    } else {
        signal[signal.len() - 1]
    };

    loop {
        if index < 0 {
            for sample in signal.iter().skip(1) {
                let value = edge - (*sample - signal[0]);
                distance += 1;
                if distance == target_distance {
                    return value;
                }
                if distance % (signal.len() - 1) == 0 {
                    edge = value;
                }
            }
            for sample in signal[..signal.len() - 1].iter().rev() {
                let value = edge + (*sample - signal[signal.len() - 1]);
                distance += 1;
                if distance == target_distance {
                    return value;
                }
                if distance % (signal.len() - 1) == 0 {
                    edge = value;
                }
            }
        } else {
            for sample in signal[..signal.len() - 1].iter().rev() {
                let value = edge - (*sample - signal[signal.len() - 1]);
                distance += 1;
                if distance == target_distance {
                    return value;
                }
                if distance % (signal.len() - 1) == 0 {
                    edge = value;
                }
            }
            for sample in signal.iter().skip(1) {
                let value = edge + (*sample - signal[0]);
                distance += 1;
                if distance == target_distance {
                    return value;
                }
                if distance % (signal.len() - 1) == 0 {
                    edge = value;
                }
            }
        }
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
}
