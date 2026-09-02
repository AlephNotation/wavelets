use std::sync::Arc;

use fearless_simd::Level as SimdLevel;

use super::butterfly::{
    self, ButterflyAnalysisCascade, ButterflySynthesisCascade, select_analysis, select_synthesis,
};
use super::layout::{Decomposition, DecompositionLayout};
use crate::plan::{PlannedDwt, PreparedFilterBank, validate_plan};
use crate::{Boundary, Dwt, Wavelet, WaveletError, WaveletNum};

/// A reusable, fixed-length multilevel DWT/IDWT plan.
///
/// The plan owns every single-level plan and coefficient offset. Allocate a
/// [`Decomposition`] and scratch buffer once, then reuse [`Self::forward_into`]
/// and [`Self::inverse_into`] without allocating.
pub struct WavedecPlan<T: WaveletNum> {
    pub(super) wavelet: Wavelet,
    pub(super) boundary: Boundary,
    pub(super) level_plans: Box<[PlannedDwt<T>]>,
    pub(super) layout: Arc<DecompositionLayout>,
    pub(super) temp_a_len: usize,
    pub(super) temp_b_len: usize,
    kernel_scratch_len: usize,
    pub(super) butterfly_analysis_cascade: Option<ButterflyAnalysisCascade<T>>,
    pub(super) butterfly_synthesis_cascade: Option<ButterflySynthesisCascade<T>>,
}

impl<T: WaveletNum> WavedecPlan<T> {
    pub(crate) fn new(
        signal_len: usize,
        wavelet: &Wavelet,
        boundary: Boundary,
        levels: usize,
        filters: PreparedFilterBank<T>,
        simd_level: SimdLevel,
    ) -> Result<Self, WaveletError> {
        let mut input_lengths = Vec::with_capacity(levels.max(1));
        let mut level_plans = Vec::with_capacity(levels);
        let mut current_len = signal_len;
        for _ in 0..levels {
            input_lengths.push(current_len);
            validate_plan(current_len, boundary)?;
            let plan = PlannedDwt::new(current_len, boundary, filters.clone(), simd_level);
            current_len = plan.coeff_len();
            level_plans.push(plan);
        }
        if levels == 0 {
            input_lengths.push(signal_len);
        }

        let coefficient_lengths: Vec<_> = level_plans.iter().map(Dwt::coeff_len).collect();
        let kernel_scratch_len = level_plans.iter().map(Dwt::scratch_len).max().unwrap_or(0);
        let butterfly_analysis_cascade = select_analysis(&level_plans, simd_level);
        let butterfly_synthesis_cascade = select_synthesis(&level_plans, simd_level);
        let conventional_temp_a_len = if levels >= 2 {
            coefficient_lengths.first().copied().unwrap_or(0)
        } else {
            0
        };
        let conventional_temp_b_len = if levels >= 3 {
            coefficient_lengths.get(1).copied().unwrap_or(0)
        } else {
            0
        };
        let fused_temp_a_len = if levels >= 4 {
            coefficient_lengths.get(1).copied().unwrap_or(0)
        } else {
            0
        };
        let fused_temp_b_len = if levels >= 6 {
            coefficient_lengths.get(3).copied().unwrap_or(0)
        } else {
            0
        };
        let fully_fused =
            butterfly_analysis_cascade.is_some() && butterfly_synthesis_cascade.is_some();
        let (temp_a_len, temp_b_len) = if fully_fused {
            (fused_temp_a_len, fused_temp_b_len)
        } else {
            (conventional_temp_a_len, conventional_temp_b_len)
        };

        Ok(Self {
            wavelet: wavelet.clone(),
            boundary,
            level_plans: level_plans.into_boxed_slice(),
            layout: Arc::new(DecompositionLayout::new(
                input_lengths,
                &coefficient_lengths,
            )),
            temp_a_len,
            temp_b_len,
            kernel_scratch_len,
            butterfly_analysis_cascade,
            butterfly_synthesis_cascade,
        })
    }

    /// Returns the input and reconstructed signal length fixed by this plan.
    pub fn signal_len(&self) -> usize {
        self.layout.input_lengths[0]
    }

    /// Returns the number of decomposition levels.
    pub fn levels(&self) -> usize {
        self.level_plans.len()
    }

    /// Returns the total number of coefficients across all bands.
    pub fn coeff_len(&self) -> usize {
        self.layout.buffer_len
    }

    /// Returns the minimum scratch-buffer length.
    pub fn scratch_len(&self) -> usize {
        self.temp_a_len + self.temp_b_len + self.kernel_scratch_len
    }

    /// Returns the boundary mode fixed by this plan.
    pub fn boundary(&self) -> Boundary {
        self.boundary
    }

    /// Returns the wavelet fixed by this plan.
    pub fn wavelet(&self) -> &Wavelet {
        &self.wavelet
    }

    /// Allocates an empty decomposition with this plan's exact band layout.
    pub fn allocate_decomposition(&self) -> Decomposition<T> {
        Decomposition {
            buffer: vec![T::zero(); self.coeff_len()],
            layout: self.layout.clone(),
            wavelet: self.wavelet.clone(),
            boundary: self.boundary,
        }
    }

    /// Allocates and computes a multilevel decomposition.
    ///
    /// # Panics
    ///
    /// Panics when `signal.len()` differs from [`Self::signal_len`].
    pub fn forward(&self, signal: &[T]) -> Decomposition<T> {
        let mut decomposition = self.allocate_decomposition();
        let mut scratch = vec![T::zero(); self.scratch_len()];
        self.forward_into(signal, &mut decomposition, &mut scratch);
        decomposition
    }

    /// Allocates and reconstructs the original signal.
    ///
    /// # Panics
    ///
    /// Panics when the decomposition's filter bank, boundary mode, band layout,
    /// or coefficient-buffer length does not match this plan.
    pub fn inverse(&self, decomposition: &Decomposition<T>) -> Vec<T> {
        let mut output = vec![T::zero(); self.signal_len()];
        let mut scratch = vec![T::zero(); self.scratch_len()];
        self.inverse_into(decomposition, &mut output, &mut scratch);
        output
    }

    /// Computes a multilevel decomposition without allocating.
    ///
    /// # Panics
    ///
    /// Panics when a buffer length or the decomposition layout does not match
    /// this plan.
    pub fn forward_into(
        &self,
        signal: &[T],
        decomposition: &mut Decomposition<T>,
        scratch: &mut [T],
    ) {
        assert_eq!(signal.len(), self.signal_len(), "incorrect signal length");
        self.assert_compatible(decomposition);
        assert!(
            scratch.len() >= self.scratch_len(),
            "scratch buffer is too small"
        );

        if self.levels() == 0 {
            decomposition.buffer.copy_from_slice(signal);
            return;
        }

        if let Some(cascade) = self.butterfly_analysis_cascade {
            butterfly::forward(self, cascade, signal, decomposition, scratch);
            return;
        }

        let scratch = &mut scratch[..self.scratch_len()];
        let (temp_a, scratch) = scratch.split_at_mut(self.temp_a_len);
        let (temp_b, kernel_scratch) = scratch.split_at_mut(self.temp_b_len);
        let last_level = self.levels() - 1;

        for (level, plan) in self.level_plans.iter().enumerate() {
            let detail_range = self.layout.details[level].clone();
            let plan_scratch = &mut kernel_scratch[..plan.scratch_len()];
            if level == last_level {
                let approx_end = self.layout.approx.end;
                let (approx, remaining) = decomposition.buffer.split_at_mut(approx_end);
                let detail =
                    &mut remaining[detail_range.start - approx_end..detail_range.end - approx_end];
                match level {
                    0 => plan.forward_into(signal, approx, detail, plan_scratch),
                    level if level % 2 == 1 => plan.forward_into(
                        &temp_a[..plan.signal_len()],
                        approx,
                        detail,
                        plan_scratch,
                    ),
                    _ => plan.forward_into(
                        &temp_b[..plan.signal_len()],
                        approx,
                        detail,
                        plan_scratch,
                    ),
                }
            } else {
                let detail = &mut decomposition.buffer[detail_range];
                match level {
                    0 => plan.forward_into(
                        signal,
                        &mut temp_a[..plan.coeff_len()],
                        detail,
                        plan_scratch,
                    ),
                    level if level % 2 == 1 => plan.forward_into(
                        &temp_a[..plan.signal_len()],
                        &mut temp_b[..plan.coeff_len()],
                        detail,
                        plan_scratch,
                    ),
                    _ => plan.forward_into(
                        &temp_b[..plan.signal_len()],
                        &mut temp_a[..plan.coeff_len()],
                        detail,
                        plan_scratch,
                    ),
                }
            }
        }
    }

    /// Reconstructs the original signal without allocating.
    ///
    /// # Panics
    ///
    /// Panics when a buffer length or the decomposition layout does not match
    /// this plan.
    pub fn inverse_into(
        &self,
        decomposition: &Decomposition<T>,
        output: &mut [T],
        scratch: &mut [T],
    ) {
        self.assert_compatible(decomposition);
        assert_eq!(output.len(), self.signal_len(), "incorrect output length");
        assert!(
            scratch.len() >= self.scratch_len(),
            "scratch buffer is too small"
        );

        if self.levels() == 0 {
            output.copy_from_slice(decomposition.approx());
            return;
        }

        if let Some(cascade) = self.butterfly_synthesis_cascade {
            butterfly::inverse(self, cascade, decomposition, output, scratch);
            return;
        }

        let scratch = &mut scratch[..self.scratch_len()];
        let (temp_a, scratch) = scratch.split_at_mut(self.temp_a_len);
        let (temp_b, kernel_scratch) = scratch.split_at_mut(self.temp_b_len);
        let last_level = self.levels() - 1;

        for level in (0..self.levels()).rev() {
            let plan = &self.level_plans[level];
            let detail = decomposition.detail(level + 1);
            let plan_scratch = &mut kernel_scratch[..plan.scratch_len()];
            if level == last_level {
                if level == 0 {
                    plan.inverse_into(decomposition.approx(), detail, output, plan_scratch);
                } else if level % 2 == 1 {
                    plan.inverse_into(
                        decomposition.approx(),
                        detail,
                        &mut temp_a[..plan.signal_len()],
                        plan_scratch,
                    );
                } else {
                    plan.inverse_into(
                        decomposition.approx(),
                        detail,
                        &mut temp_b[..plan.signal_len()],
                        plan_scratch,
                    );
                }
            } else if level == 0 {
                plan.inverse_into(&temp_a[..plan.coeff_len()], detail, output, plan_scratch);
            } else if level % 2 == 1 {
                plan.inverse_into(
                    &temp_b[..plan.coeff_len()],
                    detail,
                    &mut temp_a[..plan.signal_len()],
                    plan_scratch,
                );
            } else {
                plan.inverse_into(
                    &temp_a[..plan.coeff_len()],
                    detail,
                    &mut temp_b[..plan.signal_len()],
                    plan_scratch,
                );
            }
        }
    }

    fn assert_compatible(&self, decomposition: &Decomposition<T>) {
        assert!(
            decomposition.wavelet.has_same_filter_bank(&self.wavelet),
            "decomposition filter bank does not match plan"
        );
        assert_eq!(
            decomposition.boundary, self.boundary,
            "decomposition boundary does not match plan"
        );
        assert!(
            Arc::ptr_eq(&decomposition.layout, &self.layout) || decomposition.layout == self.layout,
            "decomposition layout does not match plan"
        );
        assert_eq!(
            decomposition.buffer.len(),
            self.coeff_len(),
            "incorrect coefficient buffer length"
        );
    }
}
