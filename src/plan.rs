use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Weak};

use crate::decomposition::{Level, WavedecPlan, resolve_levels};
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
    marker: PhantomData<T>,
}

impl<T: WaveletNum> DwtPlanner<T> {
    /// Constructs an empty planner.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            multilevel_cache: HashMap::new(),
            marker: PhantomData,
        }
    }

    /// Plans a one-level transform for signals of exactly `len` samples.
    ///
    /// Planning validates the boundary/length combination and precomputes all
    /// edge-extension and synthesis indices. Repeated identical requests reuse
    /// the same live plan.
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

        let plan: Arc<dyn Dwt<T>> = Arc::new(ScalarPlan::new(len, wavelet, boundary));
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

#[derive(Clone, Copy, Debug)]
struct SynthesisTerm<T> {
    coefficient: usize,
    low: T,
    high: T,
}

#[derive(Debug)]
struct ScalarPlan<T> {
    signal_len: usize,
    coeff_len: usize,
    dec_lo: Box<[T]>,
    dec_hi: Box<[T]>,
    analysis: Box<[AnalysisOutput]>,
    synthesis: Box<[Box<[SynthesisTerm<T>]>]>,
}

impl<T: WaveletNum> ScalarPlan<T> {
    fn new(signal_len: usize, wavelet: &Wavelet, boundary: Boundary) -> Self {
        let filter_len = wavelet.filter_len();
        let coeff_len = coefficient_len(signal_len, filter_len, boundary);
        let dec_lo: Box<[_]> = wavelet.dec_lo().iter().copied().map(T::from_f64).collect();
        let dec_hi: Box<[_]> = wavelet.dec_hi().iter().copied().map(T::from_f64).collect();
        let analysis = build_analysis(signal_len, coeff_len, filter_len, boundary);
        let synthesis = build_synthesis::<T>(
            signal_len,
            coeff_len,
            boundary,
            wavelet.rec_lo(),
            wavelet.rec_hi(),
        );
        Self {
            signal_len,
            coeff_len,
            dec_lo,
            dec_hi,
            analysis,
            synthesis,
        }
    }
}

impl<T: WaveletNum> Dwt<T> for ScalarPlan<T> {
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

        for (sample, terms) in out.iter_mut().zip(self.synthesis.iter()) {
            let mut value = T::zero();
            for term in terms.iter() {
                value += term.low * approx[term.coefficient];
                value += term.high * detail[term.coefficient];
            }
            *sample = value;
        }
    }
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

fn build_synthesis<T: WaveletNum>(
    signal_len: usize,
    coeff_len: usize,
    boundary: Boundary,
    rec_lo: &[f64],
    rec_hi: &[f64],
) -> Box<[Box<[SynthesisTerm<T>]>]> {
    let filter_len = rec_lo.len();
    let mut outputs: Vec<Vec<SynthesisTerm<T>>> = (0..signal_len).map(|_| Vec::new()).collect();

    if boundary == Boundary::Periodization {
        let periodic_len = 2 * coeff_len;
        let shift = (filter_len / 2 - 1) % periodic_len;
        for coefficient in 0..coeff_len {
            for tap in 0..filter_len {
                let unshifted = (2 * coefficient + tap) % periodic_len;
                let output = (unshifted + periodic_len - shift) % periodic_len;
                if output < signal_len {
                    outputs[output].push(SynthesisTerm {
                        coefficient,
                        low: T::from_f64(rec_lo[tap]),
                        high: T::from_f64(rec_hi[tap]),
                    });
                }
            }
        }
    } else {
        let crop = filter_len - 2;
        for (output, terms) in outputs.iter_mut().enumerate() {
            let full_index = output + crop;
            for tap in 0..filter_len {
                if full_index >= tap {
                    let difference = full_index - tap;
                    if difference.is_multiple_of(2) {
                        let coefficient = difference / 2;
                        if coefficient < coeff_len {
                            terms.push(SynthesisTerm {
                                coefficient,
                                low: T::from_f64(rec_lo[tap]),
                                high: T::from_f64(rec_hi[tap]),
                            });
                        }
                    }
                }
            }
        }
    }

    outputs.into_iter().map(Vec::into_boxed_slice).collect()
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
