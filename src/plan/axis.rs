use std::mem::size_of;

use fearless_simd::Level as SimdLevel;

use crate::WaveletNum;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::num::forward_axis_fused4_simd;
#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
use crate::num::forward_axis_fused8_simd;
use crate::num::{forward_axis_simd, inverse_axis_batched_simd, inverse_axis_simd, mul_add};
use crate::simd::{AxisAnalysis, AxisSynthesis};

use super::analysis::{EdgePlan, analyze_edge_row};
use super::synthesis::increment_wrapping;
use super::{Dwt, PlannedDwt};

// Below 24 taps, reduced coefficient loads do not repay the batched kernel's
// extra bookkeeping on the supported SIMD backends.
const MIN_BATCHED_AXIS_HALF_FILTER_LEN: usize = 12;

#[derive(Clone, Copy, Debug)]
pub(super) struct AxisGeometry {
    outer: usize,
    inner: usize,
}

impl AxisGeometry {
    pub(super) fn new(outer: usize, inner: usize) -> Self {
        Self { outer, inner }
    }

    fn buffer_len(self, axis: usize) -> usize {
        self.outer
            .checked_mul(axis)
            .and_then(|value| value.checked_mul(self.inner))
            .expect("axis buffer length overflow")
    }
}

#[derive(Debug)]
pub(super) struct AxisPlan {
    synthesis_kernel: AxisSynthesisKernel,
}

impl AxisPlan {
    pub(super) fn new(signal_len: usize, filter_len: usize, periodized: bool) -> Self {
        Self {
            synthesis_kernel: AxisSynthesisKernel::select(signal_len, filter_len, periodized),
        }
    }

    pub(super) fn scratch_len<T: WaveletNum>(
        &self,
        plan: &PlannedDwt<T>,
        geometry: AxisGeometry,
    ) -> usize {
        if let Some(batch) = AxisRowBatch::select(
            plan.simd_level,
            plan.filters.filter_len,
            size_of::<T>(),
            geometry.outer,
            geometry.inner,
        ) {
            batch.scratch_len(plan.signal_len, plan.coeff_len)
        } else if geometry.inner == 1 {
            plan.scratch_len()
        } else {
            0
        }
    }

    pub(super) fn forward_into<T: WaveletNum>(
        &self,
        plan: &PlannedDwt<T>,
        signal: &[T],
        geometry: AxisGeometry,
        approx: &mut [T],
        detail: &mut [T],
        scratch: &mut [T],
    ) {
        assert_eq!(
            signal.len(),
            geometry.buffer_len(plan.signal_len),
            "incorrect axis input length"
        );
        let output_len = geometry.buffer_len(plan.coeff_len);
        assert_eq!(
            approx.len(),
            output_len,
            "incorrect axis approximation length"
        );
        assert_eq!(detail.len(), output_len, "incorrect axis detail length");
        assert!(
            scratch.len() >= self.scratch_len(plan, geometry),
            "scratch buffer is too small"
        );

        if let Some(batch) = AxisRowBatch::select(
            plan.simd_level,
            plan.filters.filter_len,
            size_of::<T>(),
            geometry.outer,
            geometry.inner,
        ) {
            analyze_packed_rows(plan, batch, signal, geometry.outer, approx, detail, scratch);
            return;
        }

        if geometry.inner == 1 {
            for outer_index in 0..geometry.outer {
                let signal_start = outer_index * plan.signal_len;
                let output_start = outer_index * plan.coeff_len;
                plan.forward_into(
                    &signal[signal_start..signal_start + plan.signal_len],
                    &mut approx[output_start..output_start + plan.coeff_len],
                    &mut detail[output_start..output_start + plan.coeff_len],
                    scratch,
                );
            }
            return;
        }

        analyze_batched_axis(plan, signal, geometry.outer, geometry.inner, approx, detail);
    }

    pub(super) fn inverse_into<T: WaveletNum>(
        &self,
        plan: &PlannedDwt<T>,
        approx: &[T],
        detail: &[T],
        geometry: AxisGeometry,
        out: &mut [T],
        scratch: &mut [T],
    ) {
        let coefficient_len = geometry.buffer_len(plan.coeff_len);
        assert_eq!(
            approx.len(),
            coefficient_len,
            "incorrect axis approximation length"
        );
        assert_eq!(
            detail.len(),
            coefficient_len,
            "incorrect axis detail length"
        );
        assert_eq!(
            out.len(),
            geometry.buffer_len(plan.signal_len),
            "incorrect axis output length"
        );
        let required_scratch = if geometry.inner == 1 {
            plan.scratch_len()
        } else {
            0
        };
        assert!(
            scratch.len() >= required_scratch,
            "scratch buffer is too small"
        );

        if geometry.inner == 1 {
            for outer_index in 0..geometry.outer {
                let coeff_start = outer_index * plan.coeff_len;
                let output_start = outer_index * plan.signal_len;
                plan.inverse_into(
                    &approx[coeff_start..coeff_start + plan.coeff_len],
                    &detail[coeff_start..coeff_start + plan.coeff_len],
                    &mut out[output_start..output_start + plan.signal_len],
                    scratch,
                );
            }
            return;
        }

        let (rec_lo, rec_hi) = plan.synthesis_filters();
        let synthesis = AxisSynthesis {
            approx,
            detail,
            rec_lo,
            rec_hi,
            signal_len: plan.signal_len,
            coeff_len: plan.coeff_len,
            outer: geometry.outer,
            inner: geometry.inner,
            periodized_initial: plan
                .periodized_synthesis
                .as_ref()
                .map(|layout| layout.initial_coefficient),
            periodized_phases_are_swapped: plan
                .periodized_synthesis
                .as_ref()
                .is_some_and(|layout| layout.phases_are_swapped),
        };
        let vectorized = match self.synthesis_kernel {
            AxisSynthesisKernel::Direct => inverse_axis_simd(plan.simd_level, synthesis, out),
            AxisSynthesisKernel::Batched => {
                inverse_axis_batched_simd(plan.simd_level, synthesis, out)
            }
        };
        synthesize_axis_tail(
            plan,
            approx,
            detail,
            geometry.outer,
            geometry.inner,
            vectorized,
            out,
        );
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AxisAnalysisKernel {
    Direct,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Fused4,
    #[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
    Fused8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct AxisRowBatch {
    pub(super) width: usize,
}

impl AxisRowBatch {
    fn vector_geometry(level: SimdLevel, sample_size: usize) -> Option<(usize, usize)> {
        let level = level.__dispatch_target();
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if level.as_avx512().is_some() {
                return Some((64 / sample_size, 32));
            }
            if level.as_avx2().is_some() {
                let minimum_filter_len = if sample_size == size_of::<f32>() {
                    32
                } else {
                    20
                };
                return Some((32 / sample_size, minimum_filter_len));
            }
        }
        #[cfg(target_arch = "aarch64")]
        if level.as_neon().is_some() {
            let minimum_filter_len = if sample_size == size_of::<f32>() {
                32
            } else {
                48
            };
            return Some((16 / sample_size, minimum_filter_len));
        }
        let _ = level;
        let _ = sample_size;
        None
    }

    pub(super) fn select(
        level: SimdLevel,
        filter_len: usize,
        sample_size: usize,
        outer: usize,
        inner: usize,
    ) -> Option<Self> {
        if inner != 1 {
            return None;
        }

        let (width, minimum_filter_len) = Self::vector_geometry(level, sample_size)?;

        if filter_len < minimum_filter_len {
            return None;
        }

        // A tile spans several vectors to amortize packing and executor setup.
        // It remains small enough that the db38 working set stays cache-local.
        let width = width * 8;
        (outer >= width).then_some(Self { width })
    }

    pub(super) fn scratch_len(self, signal_len: usize, coeff_len: usize) -> usize {
        signal_len
            .checked_add(
                coeff_len
                    .checked_mul(2)
                    .expect("axis scratch length overflow"),
            )
            .and_then(|per_lane| per_lane.checked_mul(self.width))
            .expect("axis scratch length overflow")
    }
}

impl AxisAnalysisKernel {
    pub(super) fn select(level: SimdLevel, filter_len: usize, sample_size: usize) -> Self {
        let level = level.__dispatch_target();

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if level.as_avx512().is_some() {
                let minimum = if sample_size == size_of::<f32>() {
                    24
                } else {
                    16
                };
                if filter_len >= minimum {
                    return Self::Fused8;
                }
            } else if level.as_avx2().is_some() {
                let minimum = if sample_size == size_of::<f32>() {
                    32
                } else {
                    16
                };
                if filter_len >= minimum {
                    return Self::Fused4;
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        if level.as_neon().is_some() && filter_len >= 48 {
            return Self::Fused8;
        }

        let _ = level;
        let _ = filter_len;
        let _ = sample_size;
        Self::Direct
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AxisSynthesisKernel {
    Direct,
    Batched,
}

impl AxisSynthesisKernel {
    pub(super) fn select(signal_len: usize, filter_len: usize, periodized: bool) -> Self {
        if !periodized
            && filter_len / 2 >= MIN_BATCHED_AXIS_HALF_FILTER_LEN
            && signal_len.div_ceil(2) >= 2
        {
            Self::Batched
        } else {
            Self::Direct
        }
    }
}

pub(super) fn analyze_packed_rows<T: WaveletNum>(
    plan: &PlannedDwt<T>,
    batch: AxisRowBatch,
    signal: &[T],
    outer: usize,
    approx: &mut [T],
    detail: &mut [T],
    scratch: &mut [T],
) {
    let packed_signal_len = plan
        .signal_len
        .checked_mul(batch.width)
        .expect("axis scratch length overflow");
    let packed_band_len = plan
        .coeff_len
        .checked_mul(batch.width)
        .expect("axis scratch length overflow");
    let (packed_signal, scratch) = scratch.split_at_mut(packed_signal_len);
    let (packed_approx, packed_detail) = scratch.split_at_mut(packed_band_len);
    let packed_detail = &mut packed_detail[..packed_band_len];

    for first_row in (0..outer).step_by(batch.width) {
        let rows = (outer - first_row).min(batch.width);
        if rows < batch.width {
            packed_signal.fill(T::zero());
        }

        for sample in 0..plan.signal_len {
            let packed = &mut packed_signal[sample * batch.width..(sample + 1) * batch.width];
            for lane in 0..rows {
                packed[lane] = signal[(first_row + lane) * plan.signal_len + sample];
            }
        }

        analyze_batched_axis(
            plan,
            packed_signal,
            1,
            batch.width,
            packed_approx,
            packed_detail,
        );

        for coefficient in 0..plan.coeff_len {
            let packed_start = coefficient * batch.width;
            for lane in 0..rows {
                let output = (first_row + lane) * plan.coeff_len + coefficient;
                approx[output] = packed_approx[packed_start + lane];
                detail[output] = packed_detail[packed_start + lane];
            }
        }
    }
}

#[inline(never)]
pub(super) fn analyze_batched_axis<T: WaveletNum>(
    plan: &PlannedDwt<T>,
    signal: &[T],
    outer: usize,
    inner: usize,
    approx: &mut [T],
    detail: &mut [T],
) {
    let (dec_lo, dec_hi) = plan.filters.analysis();
    let (interior_first_newest, interior_len) =
        plan.analysis.interior.as_ref().map_or((0, 0), |interior| {
            (interior.first_newest, interior.output_len)
        });
    let analysis = AxisAnalysis {
        signal,
        dec_lo,
        dec_hi,
        edge_row_offsets: &plan.analysis.edges.row_offsets,
        edge_terms: &plan.analysis.edges.terms,
        signal_len: plan.signal_len,
        coeff_len: plan.coeff_len,
        outer,
        inner,
        prefix_len: plan.analysis.prefix_len,
        interior_first_newest,
        interior_len,
    };
    let axis_analysis_kernel =
        AxisAnalysisKernel::select(plan.simd_level, dec_lo.len(), size_of::<T>());
    let vectorized = match axis_analysis_kernel {
        AxisAnalysisKernel::Direct => forward_axis_simd(plan.simd_level, analysis, approx, detail),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        AxisAnalysisKernel::Fused4 => {
            forward_axis_fused4_simd(plan.simd_level, analysis, approx, detail)
        }
        #[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
        AxisAnalysisKernel::Fused8 => {
            forward_axis_fused8_simd(plan.simd_level, analysis, approx, detail)
        }
    };
    analyze_axis_tail(plan, signal, outer, inner, vectorized, approx, detail);
}

pub(super) fn analyze_axis_tail<T: WaveletNum>(
    plan: &PlannedDwt<T>,
    signal: &[T],
    outer: usize,
    inner: usize,
    first_lane: usize,
    approx: &mut [T],
    detail: &mut [T],
) {
    let (dec_lo, dec_hi) = plan.filters.analysis();
    let prefix_len = plan.analysis.prefix_len;
    let interior_len = plan
        .analysis
        .interior
        .as_ref()
        .map_or(0, |interior| interior.output_len);
    let interior_end = prefix_len + interior_len;

    for outer_index in 0..outer {
        let signal_start = outer_index * plan.signal_len * inner;
        let output_start = outer_index * plan.coeff_len * inner;
        for lane in first_lane..inner {
            for output in 0..prefix_len {
                let (low, high) = analyze_axis_edge_scalar(
                    &signal[signal_start..],
                    inner,
                    lane,
                    &plan.analysis.edges,
                    output,
                    dec_lo,
                    dec_hi,
                );
                approx[output_start + output * inner + lane] = low;
                detail[output_start + output * inner + lane] = high;
            }

            if let Some(interior) = &plan.analysis.interior {
                for interior_output in 0..interior.output_len {
                    let newest = interior.first_newest + 2 * interior_output;
                    let mut low = T::zero();
                    let mut high = T::zero();
                    for tap in 0..dec_lo.len() {
                        let sample = signal[signal_start + (newest - tap) * inner + lane];
                        low = mul_add(sample, dec_lo[tap], low);
                        high = mul_add(sample, dec_hi[tap], high);
                    }
                    let output = prefix_len + interior_output;
                    approx[output_start + output * inner + lane] = low;
                    detail[output_start + output * inner + lane] = high;
                }
            }

            for output in interior_end..plan.coeff_len {
                let edge_row = prefix_len + output - interior_end;
                let (low, high) = analyze_axis_edge_scalar(
                    &signal[signal_start..],
                    inner,
                    lane,
                    &plan.analysis.edges,
                    edge_row,
                    dec_lo,
                    dec_hi,
                );
                approx[output_start + output * inner + lane] = low;
                detail[output_start + output * inner + lane] = high;
            }
        }
    }
}

fn analyze_axis_edge_scalar<T: WaveletNum>(
    signal: &[T],
    inner: usize,
    lane: usize,
    edges: &EdgePlan<T>,
    row: usize,
    dec_lo: &[T],
    dec_hi: &[T],
) -> (T, T) {
    analyze_edge_row(edges, row, dec_lo, dec_hi, |input| {
        signal[input * inner + lane]
    })
}

pub(super) fn synthesize_axis_tail<T: WaveletNum>(
    plan: &PlannedDwt<T>,
    approx: &[T],
    detail: &[T],
    outer: usize,
    inner: usize,
    first_lane: usize,
    out: &mut [T],
) {
    let (rec_lo, rec_hi) = plan.synthesis_filters();
    let half_filter_len = rec_lo.len() / 2;
    let (first_lo, second_lo) = rec_lo.split_at(half_filter_len);
    let (first_hi, second_hi) = rec_hi.split_at(half_filter_len);
    let output_pairs = plan.signal_len.div_ceil(2);

    for outer_index in 0..outer {
        let coeff_start = outer_index * plan.coeff_len * inner;
        let output_start = outer_index * plan.signal_len * inner;
        for lane in first_lane..inner {
            for pair in 0..output_pairs {
                let first_coefficient = plan
                    .periodized_synthesis
                    .as_ref()
                    .map_or(pair + half_filter_len - 1, |layout| {
                        (layout.initial_coefficient + pair) % plan.coeff_len
                    });
                let second_coefficient = if plan
                    .periodized_synthesis
                    .as_ref()
                    .is_some_and(|layout| layout.phases_are_swapped)
                {
                    increment_wrapping(first_coefficient, plan.coeff_len)
                } else {
                    first_coefficient
                };
                let mut first = T::zero();
                let mut second = T::zero();
                for tap in 0..half_filter_len {
                    let first_index = if plan.periodized_synthesis.is_some() {
                        (first_coefficient + plan.coeff_len - tap % plan.coeff_len) % plan.coeff_len
                    } else {
                        first_coefficient - tap
                    };
                    let second_index = if plan.periodized_synthesis.is_some() {
                        (second_coefficient + plan.coeff_len - tap % plan.coeff_len)
                            % plan.coeff_len
                    } else {
                        second_coefficient - tap
                    };
                    let first_offset = coeff_start + first_index * inner + lane;
                    let second_offset = coeff_start + second_index * inner + lane;
                    first = mul_add(approx[first_offset], first_lo[tap], first);
                    first = mul_add(detail[first_offset], first_hi[tap], first);
                    second = mul_add(approx[second_offset], second_lo[tap], second);
                    second = mul_add(detail[second_offset], second_hi[tap], second);
                }
                let first_output = output_start + 2 * pair * inner + lane;
                out[first_output] = first;
                if 2 * pair + 1 < plan.signal_len {
                    out[first_output + inner] = second;
                }
            }
        }
    }
}
