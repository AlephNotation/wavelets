use std::mem::size_of;
use std::sync::Arc;

use crate::num::{is_finite, mul_add};
use crate::{Boundary, WaveletNum};

use super::analysis::for_each_extension_term;

// The scan costs more than the existing SIMD kernel below this support on the
// measured NEON backend. Keeping the cutoff algebraic lets equivalent custom
// banks qualify without coupling execution to built-in wavelet names.
const MIN_ANNIHILATOR_FILTER_LEN_F64: usize = 76;
pub(super) const MIN_ANNIHILATOR_FILTER_LEN_F32: usize = 102;
const ANNIHILATOR_BASE_FILTER_LEN: usize = 64;
const ANNIHILATOR_EVENT_COST_SCALE_F64: usize = 6;
const ANNIHILATOR_EVENT_COST_SCALE_F32: usize = 12;

#[derive(Debug)]
pub(super) struct AnnihilatorFilter<T> {
    low_base: T,
    high_base: T,
    low_correction: Box<[T]>,
    high_correction: Box<[T]>,
}

impl<T: WaveletNum> AnnihilatorFilter<T> {
    pub(super) fn new(dec_lo: &[T], dec_hi: &[T]) -> Option<Self> {
        let minimum_filter_len = if size_of::<T>() == size_of::<f32>() {
            MIN_ANNIHILATOR_FILTER_LEN_F32
        } else {
            MIN_ANNIHILATOR_FILTER_LEN_F64
        };
        if dec_lo.len() < minimum_filter_len {
            return None;
        }
        let (low_base, low_correction) = factor_degree_zero(dec_lo);
        let (high_base, high_correction) = factor_degree_zero(dec_hi);
        Some(Self {
            low_base,
            high_base,
            low_correction: low_correction.into_boxed_slice(),
            high_correction: high_correction.into_boxed_slice(),
        })
    }
}

#[derive(Debug)]
pub(super) struct AnnihilatorAnalysis<T> {
    filter: Arc<AnnihilatorFilter<T>>,
    boundary: Boundary,
    phase: isize,
    first_extended_index: isize,
    extension_len: usize,
    maximum_events: usize,
}

impl<T: WaveletNum> AnnihilatorAnalysis<T> {
    pub(super) fn new(
        signal_len: usize,
        coeff_len: usize,
        boundary: Boundary,
        filter: Arc<AnnihilatorFilter<T>>,
    ) -> Self {
        let filter_len = filter.low_correction.len() + 1;
        let phase = if boundary == Boundary::Periodization {
            (filter_len / 2) as isize
        } else {
            1
        };
        let first_extended_index = phase - (filter_len - 1) as isize;
        let extension_len = 2 * coeff_len - 2 + filter_len;
        // A correction event touches about half the filter in both bands. This
        // conservative M4-derived budget admits the measured db38/coif17 win
        // regions while rejecting db20-like marginal cases entirely above.
        // f32 direct SIMD processes twice as many samples per vector while
        // scalar correction scattering does not, so each event must be
        // charged twice as heavily as f64.
        let event_cost_scale = if size_of::<T>() == size_of::<f32>() {
            ANNIHILATOR_EVENT_COST_SCALE_F32
        } else {
            ANNIHILATOR_EVENT_COST_SCALE_F64
        };
        let maximum_events = ((signal_len as u128
            * filter_len.saturating_sub(ANNIHILATOR_BASE_FILTER_LEN) as u128)
            / (event_cost_scale * filter_len) as u128) as usize;
        Self {
            filter,
            boundary,
            phase,
            first_extended_index,
            extension_len,
            maximum_events,
        }
    }

    pub(super) fn should_execute(&self, signal: &[T]) -> bool {
        let mut events = 0;
        let mut previous = signal[0];
        if !is_finite(previous) {
            return false;
        }
        for &current in &signal[1..] {
            if !is_finite(current) {
                return false;
            }
            let amplitude = current - previous;
            if !is_finite(amplitude) {
                return false;
            }
            if amplitude != T::zero() {
                events += 1;
                if events > self.maximum_events {
                    return false;
                }
            }
            previous = current;
        }

        // The finite extension can introduce additional jumps even when the
        // original signal is sparse. Count only the two O(filter_len) halos;
        // the interior transitions were counted above.
        if self.first_extended_index < 0 {
            let mut previous = extended_sample(signal, self.first_extended_index, self.boundary);
            if !is_finite(previous) {
                return false;
            }
            for index in self.first_extended_index + 1..=0 {
                let current = extended_sample(signal, index, self.boundary);
                let amplitude = current - previous;
                if !is_finite(current) || !is_finite(amplitude) {
                    return false;
                }
                if amplitude != T::zero() {
                    events += 1;
                    if events > self.maximum_events {
                        return false;
                    }
                }
                previous = current;
            }
        }

        let final_extended_index = self.first_extended_index + self.extension_len as isize - 1;
        let final_signal_index = signal.len() as isize - 1;
        if final_extended_index > final_signal_index {
            let mut previous = signal[signal.len() - 1];
            for index in signal.len() as isize..=final_extended_index {
                let current = extended_sample(signal, index, self.boundary);
                let amplitude = current - previous;
                if !is_finite(current) || !is_finite(amplitude) {
                    return false;
                }
                if amplitude != T::zero() {
                    events += 1;
                    if events > self.maximum_events {
                        return false;
                    }
                }
                previous = current;
            }
        }
        true
    }

    pub(super) fn forward_into(&self, signal: &[T], approx: &mut [T], detail: &mut [T]) {
        for coefficient in 0..approx.len() {
            let base_index = self.first_extended_index + 2 * coefficient as isize;
            let sample = extended_sample(signal, base_index, self.boundary);
            approx[coefficient] = self.filter.low_base * sample;
            detail[coefficient] = self.filter.high_base * sample;
        }

        let mut previous = extended_sample(signal, self.first_extended_index, self.boundary);
        for offset in 1..self.extension_len {
            let event = self.first_extended_index + offset as isize;
            let current = extended_sample(signal, event, self.boundary);
            let amplitude = current - previous;
            if amplitude != T::zero() {
                self.scatter_event(event, amplitude, approx, detail);
            }
            previous = current;
        }
    }

    #[inline]
    fn scatter_event(&self, event: isize, amplitude: T, approx: &mut [T], detail: &mut [T]) {
        let first_tap = (self.phase - event).rem_euclid(2) as usize;
        for tap in (first_tap..self.filter.low_correction.len()).step_by(2) {
            let output_offset = event + tap as isize - self.phase;
            if output_offset < 0 {
                continue;
            }
            let coefficient = output_offset as usize / 2;
            if coefficient >= approx.len() {
                break;
            }
            approx[coefficient] = mul_add(
                amplitude,
                self.filter.low_correction[tap],
                approx[coefficient],
            );
            detail[coefficient] = mul_add(
                amplitude,
                self.filter.high_correction[tap],
                detail[coefficient],
            );
        }
    }
}

fn factor_degree_zero<T: WaveletNum>(filter: &[T]) -> (T, Vec<T>) {
    debug_assert!(filter.len() >= 2);
    let mut base = T::zero();
    for &tap in filter {
        base += tap;
    }
    let mut running = T::zero();
    let correction = filter[..filter.len() - 1]
        .iter()
        .map(|&tap| {
            running += tap;
            running
        })
        .collect();
    (base, correction)
}

#[inline]
fn extended_sample<T: WaveletNum>(signal: &[T], index: isize, boundary: Boundary) -> T {
    if (0..signal.len() as isize).contains(&index) {
        return signal[index as usize];
    }
    if boundary == Boundary::Smooth {
        if signal.len() == 1 {
            return signal[0];
        }
        if index < 0 {
            let distance = T::from_f64((-index) as f64);
            return signal[0] + (signal[0] - signal[1]) * distance;
        }
        let last = signal.len() - 1;
        let distance = T::from_f64((index - last as isize) as f64);
        return signal[last] + (signal[last] - signal[last - 1]) * distance;
    }
    let mut sample = T::zero();
    for_each_extension_term(index, signal.len(), boundary, |input, weight| {
        sample = mul_add(signal[input], weight, sample);
    });
    sample
}
