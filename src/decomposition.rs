use std::ops::Range;
use std::sync::Arc;

use fearless_simd::Level as SimdLevel;

use crate::num::{forward_butterfly_pair_simd, inverse_butterfly_pair_simd};
use crate::plan::{PlannedDwt, PreparedFilterBank, validate_plan};
use crate::simd::{ButterflyPairAnalysis, ButterflyPairSynthesis};
use crate::{Boundary, Dwt, DwtPlanner, Wavelet, WaveletError, WaveletNum};

/// Selects the number of levels in a multilevel decomposition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Level {
    /// Use the largest level at which at least one coefficient is unaffected by
    /// boundary extension.
    Max,
    /// Use exactly this many levels, rejecting values above [`Level::Max`].
    Exact(usize),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DecompositionLayout {
    approx: Range<usize>,
    details: Box<[Range<usize>]>,
    input_lengths: Box<[usize]>,
    buffer_len: usize,
}

impl DecompositionLayout {
    fn new(input_lengths: Vec<usize>, coefficient_lengths: &[usize]) -> Self {
        debug_assert_eq!(input_lengths.len(), coefficient_lengths.len().max(1));
        let approx_len = coefficient_lengths
            .last()
            .copied()
            .unwrap_or(input_lengths[0]);
        let mut buffer_len = approx_len;
        let mut details = vec![0..0; coefficient_lengths.len()];
        for level in (0..coefficient_lengths.len()).rev() {
            let start = buffer_len;
            buffer_len += coefficient_lengths[level];
            details[level] = start..buffer_len;
        }
        Self {
            approx: 0..approx_len,
            details: details.into_boxed_slice(),
            input_lengths: input_lengths.into_boxed_slice(),
            buffer_len,
        }
    }
}

/// An owned multilevel decomposition stored in one contiguous allocation.
///
/// The physical layout is `cA_L, cD_L, ..., cD_1`, while [`Self::detail`]
/// addresses detail bands by their natural one-based level. A decomposition
/// allocated by [`WavedecPlan::allocate_decomposition`] can be overwritten and
/// reused without allocating.
///
/// # Examples
///
/// ```
/// use wavelets::{Boundary, Level, Wavelet, wavedec};
///
/// let signal: Vec<f64> = (0..32).map(f64::from).collect();
/// let wavelet = Wavelet::haar();
/// let decomposition = wavedec(
///     &signal,
///     &wavelet,
///     Boundary::Symmetric,
///     Level::Exact(3),
/// )?;
///
/// assert_eq!(decomposition.levels(), 3);
/// assert_eq!(decomposition.detail(1).len(), 16);
/// assert_eq!(decomposition.detail(3).len(), 4);
/// assert_eq!(decomposition.bands().count(), 4);
/// # Ok::<(), wavelets::WaveletError>(())
/// ```
#[derive(Clone, Debug)]
pub struct Decomposition<T> {
    buffer: Vec<T>,
    layout: Arc<DecompositionLayout>,
    wavelet: Wavelet,
    boundary: Boundary,
}

impl<T> Decomposition<T> {
    /// Returns the number of decomposition levels.
    pub fn levels(&self) -> usize {
        self.layout.details.len()
    }

    /// Returns the coarsest approximation band, `cA_L`.
    pub fn approx(&self) -> &[T] {
        &self.buffer[self.layout.approx.clone()]
    }

    /// Returns `cD_level` for a one-based level in `1..=levels()`.
    ///
    /// # Panics
    ///
    /// Panics when `level` is zero or greater than [`Self::levels`].
    pub fn detail(&self, level: usize) -> &[T] {
        let range = self
            .layout
            .details
            .get(level.checked_sub(1).expect("detail levels are one-based"))
            .expect("detail level is out of range")
            .clone();
        &self.buffer[range]
    }

    /// Mutably returns the coarsest approximation band, `cA_L`.
    pub fn approx_mut(&mut self) -> &mut [T] {
        &mut self.buffer[self.layout.approx.clone()]
    }

    /// Mutably returns `cD_level` for a one-based level in `1..=levels()`.
    ///
    /// # Panics
    ///
    /// Panics when `level` is zero or greater than [`Self::levels`].
    pub fn detail_mut(&mut self, level: usize) -> &mut [T] {
        let range = self
            .layout
            .details
            .get(level.checked_sub(1).expect("detail levels are one-based"))
            .expect("detail level is out of range")
            .clone();
        &mut self.buffer[range]
    }

    /// Returns the original signal length.
    pub fn original_len(&self) -> usize {
        self.layout.input_lengths[0]
    }

    /// Returns the boundary mode used to create this decomposition.
    pub fn boundary(&self) -> Boundary {
        self.boundary
    }

    /// Returns the wavelet used to create this decomposition.
    pub fn wavelet(&self) -> &Wavelet {
        &self.wavelet
    }

    /// Returns the contiguous backing storage in physical band order.
    pub fn as_slice(&self) -> &[T] {
        &self.buffer
    }

    /// Mutably returns the contiguous backing storage in physical band order.
    ///
    /// The layout is `cA_L, cD_L, ..., cD_1`. Prefer [`Self::approx_mut`] and
    /// [`Self::detail_mut`] when an operation targets a particular band.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.buffer
    }

    /// Iterates over coefficient bands in PyWavelets order:
    /// `cA_L, cD_L, ..., cD_1`.
    pub fn bands(&self) -> impl Iterator<Item = &[T]> {
        std::iter::once(self.approx())
            .chain((0..self.levels()).rev().map(|level| self.detail(level + 1)))
    }
}

/// A reusable, fixed-length multilevel DWT/IDWT plan.
///
/// The plan owns every single-level plan and coefficient offset. Allocate a
/// [`Decomposition`] and scratch buffer once, then reuse [`Self::forward_into`]
/// and [`Self::inverse_into`] without allocating.
pub struct WavedecPlan<T: WaveletNum> {
    wavelet: Wavelet,
    boundary: Boundary,
    level_plans: Box<[PlannedDwt<T>]>,
    layout: Arc<DecompositionLayout>,
    temp_a_len: usize,
    temp_b_len: usize,
    kernel_scratch_len: usize,
    butterfly_analysis_cascade: Option<ButterflyAnalysisCascade<T>>,
    butterfly_synthesis_cascade: Option<ButterflySynthesisCascade<T>>,
}

#[derive(Clone, Copy)]
struct ButterflyAnalysisCascade<T> {
    simd_level: SimdLevel,
    low_scale: T,
    high_scale: T,
}

#[derive(Clone, Copy)]
struct ButterflySynthesisCascade<T> {
    simd_level: SimdLevel,
    low_scale: T,
    high_scale: T,
}

// Four-way lane rearrangement has a fixed cost. Sixty-four fused outputs are
// enough to amortize it on NEON; this remains conservative for wider vectors.
const MIN_FUSED_ANALYSIS_OUTPUTS: usize = 64;

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

        let coefficient_lengths: Vec<_> = level_plans.iter().map(|plan| plan.coeff_len()).collect();
        let kernel_scratch_len = level_plans
            .iter()
            .map(|plan| plan.scratch_len())
            .max()
            .unwrap_or(0);
        let paired_geometry = levels >= 2 && levels.is_multiple_of(2);
        let butterfly_analysis_cascade = if paired_geometry
            && level_plans[1].coeff_len() >= MIN_FUSED_ANALYSIS_OUTPUTS
            && level_plans
                .iter()
                .all(|plan| plan.full_butterfly_analysis().is_some())
        {
            let (low_scale, high_scale) = level_plans[0]
                .full_butterfly_analysis()
                .expect("the complete analysis cascade was checked above");
            Some(ButterflyAnalysisCascade {
                simd_level,
                low_scale,
                high_scale,
            })
        } else {
            None
        };
        let butterfly_synthesis_cascade = if paired_geometry
            && level_plans
                .iter()
                .all(|plan| plan.full_butterfly_synthesis().is_some())
        {
            let (low_scale, high_scale) = level_plans[0]
                .full_butterfly_synthesis()
                .expect("the complete synthesis cascade was checked above");
            Some(ButterflySynthesisCascade {
                simd_level,
                low_scale,
                high_scale,
            })
        } else {
            None
        };
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
        let temp_a_len =
            if butterfly_analysis_cascade.is_some() && butterfly_synthesis_cascade.is_some() {
                fused_temp_a_len
            } else {
                conventional_temp_a_len
            };
        let temp_b_len =
            if butterfly_analysis_cascade.is_some() && butterfly_synthesis_cascade.is_some() {
                fused_temp_b_len
            } else {
                conventional_temp_b_len
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
            self.forward_butterfly_cascade(cascade, signal, decomposition, scratch);
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
            self.inverse_butterfly_cascade(cascade, decomposition, output, scratch);
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

    fn forward_butterfly_cascade(
        &self,
        cascade: ButterflyAnalysisCascade<T>,
        signal: &[T],
        decomposition: &mut Decomposition<T>,
        scratch: &mut [T],
    ) {
        let scratch = &mut scratch[..self.scratch_len()];
        let (temp_a, scratch) = scratch.split_at_mut(self.temp_a_len);
        let (temp_b, _) = scratch.split_at_mut(self.temp_b_len);
        let pair_count = self.levels() / 2;

        for pair in 0..pair_count {
            let first_level = 2 * pair;
            let first_detail_range = self.layout.details[first_level].clone();
            let second_detail_range = self.layout.details[first_level + 1].clone();
            let final_pair = pair + 1 == pair_count;

            if final_pair {
                let approx_end = self.layout.approx.end;
                let (approx, details) = decomposition.buffer.split_at_mut(approx_end);
                let (first_detail, second_detail) =
                    detail_pair_mut(details, approx_end, first_detail_range, second_detail_range);
                match pair {
                    0 => {
                        forward_butterfly_pair(cascade, signal, approx, first_detail, second_detail)
                    }
                    pair if pair % 2 == 1 => forward_butterfly_pair(
                        cascade,
                        &temp_a[..4 * approx.len()],
                        approx,
                        first_detail,
                        second_detail,
                    ),
                    _ => forward_butterfly_pair(
                        cascade,
                        &temp_b[..4 * approx.len()],
                        approx,
                        first_detail,
                        second_detail,
                    ),
                }
            } else {
                let (first_detail, second_detail) = detail_pair_mut(
                    &mut decomposition.buffer,
                    0,
                    first_detail_range,
                    second_detail_range,
                );
                let output_len = second_detail.len();
                if pair == 0 {
                    forward_butterfly_pair(
                        cascade,
                        signal,
                        &mut temp_a[..output_len],
                        first_detail,
                        second_detail,
                    );
                } else if pair % 2 == 1 {
                    forward_butterfly_pair(
                        cascade,
                        &temp_a[..4 * output_len],
                        &mut temp_b[..output_len],
                        first_detail,
                        second_detail,
                    );
                } else {
                    forward_butterfly_pair(
                        cascade,
                        &temp_b[..4 * output_len],
                        &mut temp_a[..output_len],
                        first_detail,
                        second_detail,
                    );
                }
            }
        }
    }

    fn inverse_butterfly_cascade(
        &self,
        cascade: ButterflySynthesisCascade<T>,
        decomposition: &Decomposition<T>,
        output: &mut [T],
        scratch: &mut [T],
    ) {
        let scratch = &mut scratch[..self.scratch_len()];
        let (temp_a, scratch) = scratch.split_at_mut(self.temp_a_len);
        let (temp_b, _) = scratch.split_at_mut(self.temp_b_len);
        let pair_count = self.levels() / 2;

        for pair in (0..pair_count).rev() {
            let first_level = 2 * pair;
            let first_detail = decomposition.detail(first_level + 1);
            let second_detail = decomposition.detail(first_level + 2);
            let output_len = 4 * second_detail.len();

            if pair == 0 {
                let approx = if pair + 1 == pair_count {
                    decomposition.approx()
                } else if pair % 2 == 0 {
                    &temp_a[..second_detail.len()]
                } else {
                    &temp_b[..second_detail.len()]
                };
                inverse_butterfly_pair(cascade, approx, first_detail, second_detail, output);
            } else {
                let coarsest_pair = pair + 1 == pair_count;
                if pair % 2 == 1 {
                    let approx = if coarsest_pair {
                        decomposition.approx()
                    } else {
                        &temp_b[..second_detail.len()]
                    };
                    inverse_butterfly_pair(
                        cascade,
                        approx,
                        first_detail,
                        second_detail,
                        &mut temp_a[..output_len],
                    );
                } else {
                    let approx = if coarsest_pair {
                        decomposition.approx()
                    } else {
                        &temp_a[..second_detail.len()]
                    };
                    inverse_butterfly_pair(
                        cascade,
                        approx,
                        first_detail,
                        second_detail,
                        &mut temp_b[..output_len],
                    );
                }
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

fn detail_pair_mut<T>(
    buffer: &mut [T],
    buffer_offset: usize,
    first_range: Range<usize>,
    second_range: Range<usize>,
) -> (&mut [T], &mut [T]) {
    debug_assert!(second_range.end <= first_range.start);
    let second_start = second_range.start - buffer_offset;
    let first_start = first_range.start - buffer_offset;
    let (_, from_second) = buffer.split_at_mut(second_start);
    let (second_detail, after_second) = from_second.split_at_mut(second_range.len());
    let first_start = first_start - second_start - second_range.len();
    let first_detail = &mut after_second[first_start..first_start + first_range.len()];
    (first_detail, second_detail)
}

fn forward_butterfly_pair<T: WaveletNum>(
    cascade: ButterflyAnalysisCascade<T>,
    signal: &[T],
    approx: &mut [T],
    first_detail: &mut [T],
    second_detail: &mut [T],
) {
    debug_assert_eq!(signal.len(), 4 * approx.len());
    debug_assert_eq!(first_detail.len(), 2 * approx.len());
    debug_assert_eq!(second_detail.len(), approx.len());
    let vectorized = forward_butterfly_pair_simd(
        cascade.simd_level,
        ButterflyPairAnalysis {
            signal,
            first_low_scale: cascade.low_scale,
            first_high_scale: cascade.high_scale,
            second_low_scale: cascade.low_scale,
            second_high_scale: cascade.high_scale,
        },
        approx,
        first_detail,
        second_detail,
    );

    for output in vectorized..approx.len() {
        let input = 4 * output;
        let first_low = (signal[input] + signal[input + 1]) * cascade.low_scale;
        let second_low = (signal[input + 2] + signal[input + 3]) * cascade.low_scale;
        first_detail[2 * output] = (signal[input] - signal[input + 1]) * cascade.high_scale;
        first_detail[2 * output + 1] = (signal[input + 2] - signal[input + 3]) * cascade.high_scale;
        approx[output] = (first_low + second_low) * cascade.low_scale;
        second_detail[output] = (first_low - second_low) * cascade.high_scale;
    }
}

fn inverse_butterfly_pair<T: WaveletNum>(
    cascade: ButterflySynthesisCascade<T>,
    approx: &[T],
    first_detail: &[T],
    second_detail: &[T],
    out: &mut [T],
) {
    debug_assert_eq!(out.len(), 4 * approx.len());
    debug_assert_eq!(first_detail.len(), 2 * approx.len());
    debug_assert_eq!(second_detail.len(), approx.len());
    let vectorized = inverse_butterfly_pair_simd(
        cascade.simd_level,
        ButterflyPairSynthesis {
            approx,
            first_detail,
            second_detail,
            first_low_scale: cascade.low_scale,
            first_high_scale: cascade.high_scale,
            second_low_scale: cascade.low_scale,
            second_high_scale: cascade.high_scale,
        },
        out,
    );

    for input in vectorized..approx.len() {
        let second_low = approx[input] * cascade.low_scale;
        let second_high = second_detail[input] * cascade.high_scale;
        let first_approx = second_low + second_high;
        let second_approx = second_low - second_high;
        let first_low = first_approx * cascade.low_scale;
        let first_high = first_detail[2 * input] * cascade.high_scale;
        let second_low = second_approx * cascade.low_scale;
        let second_high = first_detail[2 * input + 1] * cascade.high_scale;
        let output = 4 * input;
        out[output] = first_low + first_high;
        out[output + 1] = first_low - first_high;
        out[output + 2] = second_low + second_high;
        out[output + 3] = second_low - second_high;
    }
}

/// Computes the largest decomposition level with a boundary-independent
/// coefficient, matching PyWavelets' `dwt_max_level` definition.
///
/// A `filter_len` smaller than two has no valid decomposition level and returns
/// zero.
///
/// # Examples
///
/// ```
/// use wavelets::dwt_max_level;
///
/// assert_eq!(dwt_max_level(1_000, 8), 7);
/// ```
pub fn dwt_max_level(signal_len: usize, filter_len: usize) -> usize {
    if filter_len < 2 {
        return 0;
    }
    let divisor = filter_len - 1;
    if signal_len < divisor {
        return 0;
    }
    (usize::BITS - 1 - (signal_len / divisor).leading_zeros()) as usize
}

pub(crate) fn resolve_levels(
    signal_len: usize,
    filter_len: usize,
    level: Level,
) -> Result<usize, WaveletError> {
    if signal_len == 0 {
        return Err(WaveletError::EmptySignal);
    }
    let maximum = dwt_max_level(signal_len, filter_len);
    match level {
        Level::Max => Ok(maximum),
        Level::Exact(requested) if requested <= maximum => Ok(requested),
        Level::Exact(requested) => Err(WaveletError::InvalidLevel { requested, maximum }),
    }
}

/// Computes a multilevel one-dimensional wavelet decomposition.
///
/// `Level::Exact(0)` is an identity decomposition containing only `cA_0`.
///
/// # Errors
///
/// Returns [`WaveletError::EmptySignal`] for an empty signal or
/// [`WaveletError::InvalidLevel`] when an exact level exceeds the maximum.
pub fn wavedec<T: WaveletNum>(
    signal: &[T],
    wavelet: &Wavelet,
    boundary: Boundary,
    level: Level,
) -> Result<Decomposition<T>, WaveletError> {
    let mut planner = DwtPlanner::<T>::new();
    let plan = planner.plan_wavedec(signal.len(), wavelet, boundary, level)?;
    Ok(plan.forward(signal))
}

/// Reconstructs a signal from a decomposition created by [`wavedec`].
///
/// # Errors
///
/// Returns an error if the decomposition metadata cannot produce a valid plan.
pub fn waverec<T: WaveletNum>(dec: &Decomposition<T>) -> Result<Vec<T>, WaveletError> {
    let mut planner = DwtPlanner::<T>::new();
    let plan = planner.plan_wavedec(
        dec.original_len(),
        &dec.wavelet,
        dec.boundary,
        Level::Exact(dec.levels()),
    )?;
    Ok(plan.inverse(dec))
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOUNDARIES: [Boundary; 9] = [
        Boundary::Zero,
        Boundary::Constant,
        Boundary::Symmetric,
        Boundary::Reflect,
        Boundary::Periodic,
        Boundary::Smooth,
        Boundary::Antisymmetric,
        Boundary::Antireflect,
        Boundary::Periodization,
    ];

    #[test]
    fn max_level_matches_known_values() {
        assert_eq!(dwt_max_level(1, 2), 0);
        assert_eq!(dwt_max_level(2, 2), 1);
        assert_eq!(dwt_max_level(4, 4), 0);
        assert_eq!(dwt_max_level(6, 4), 1);
        assert_eq!(dwt_max_level(12, 4), 2);
        assert_eq!(dwt_max_level(1000, 8), 7);
    }

    #[test]
    fn butterfly_cascade_selection_depends_on_algebra_and_geometry() {
        let wavelet =
            Wavelet::from_filters(&[0.5, 0.5], &[-0.25, 0.25], &[0.75, 0.75], &[0.125, -0.125])
                .unwrap();
        let mut planner = DwtPlanner::<f64>::new();

        for boundary in BOUNDARIES {
            let plan = planner
                .plan_wavedec(256, &wavelet, boundary, Level::Exact(8))
                .unwrap();
            assert!(plan.butterfly_analysis_cascade.is_some(), "{boundary:?}");
            assert!(plan.butterfly_synthesis_cascade.is_some(), "{boundary:?}");
            assert_eq!(plan.scratch_len(), 80, "{boundary:?}");
        }

        let short = planner
            .plan_wavedec(64, &wavelet, Boundary::Symmetric, Level::Exact(6))
            .unwrap();
        assert!(short.butterfly_analysis_cascade.is_none());
        assert!(short.butterfly_synthesis_cascade.is_some());
        assert_eq!(short.scratch_len(), 48);

        let odd_level_count = planner
            .plan_wavedec(64, &wavelet, Boundary::Symmetric, Level::Exact(5))
            .unwrap();
        assert!(odd_level_count.butterfly_analysis_cascade.is_none());
        assert!(odd_level_count.butterfly_synthesis_cascade.is_none());
        let edge_bearing_level = planner
            .plan_wavedec(20, &wavelet, Boundary::Symmetric, Level::Exact(4))
            .unwrap();
        assert!(edge_bearing_level.butterfly_analysis_cascade.is_none());
        assert!(edge_bearing_level.butterfly_synthesis_cascade.is_none());
    }

    #[test]
    fn custom_butterfly_cascade_matches_two_single_level_plans() {
        let wavelet =
            Wavelet::from_filters(&[0.5, 0.5], &[-0.25, 0.25], &[0.75, 0.75], &[0.125, -0.125])
                .unwrap();
        let signal: Vec<_> = (0..256)
            .map(|index| (index as f64 * 0.19).sin() + index as f64 * 0.03)
            .collect();
        let mut planner = DwtPlanner::<f64>::new();
        let first = planner
            .plan_dwt(signal.len(), &wavelet, Boundary::Symmetric)
            .unwrap();
        let second = planner
            .plan_dwt(first.coeff_len(), &wavelet, Boundary::Symmetric)
            .unwrap();
        let cascade = planner
            .plan_wavedec(signal.len(), &wavelet, Boundary::Symmetric, Level::Exact(2))
            .unwrap();
        assert_eq!(cascade.scratch_len(), 0);

        let (first_approx, first_detail) = first.forward(&signal);
        let (expected_approx, expected_second_detail) = second.forward(&first_approx);
        let actual = cascade.forward(&signal);
        assert_eq!(actual.approx(), expected_approx);
        assert_eq!(actual.detail(1), first_detail);
        assert_eq!(actual.detail(2), expected_second_detail);

        let expected_first_approx = second.inverse(actual.approx(), actual.detail(2));
        let expected_signal = first.inverse(&expected_first_approx, actual.detail(1));
        assert_eq!(cascade.inverse(&actual), expected_signal);
    }

    #[test]
    fn decomposition_uses_natural_detail_level_numbers() {
        let signal: Vec<_> = (0..16).map(f64::from).collect();
        let wavelet = Wavelet::haar();
        let dec = wavedec(&signal, &wavelet, Boundary::Symmetric, Level::Exact(3)).unwrap();
        assert_eq!(dec.levels(), 3);
        assert_eq!(dec.detail(1).len(), 8);
        assert_eq!(dec.detail(2).len(), 4);
        assert_eq!(dec.detail(3).len(), 2);
        assert_eq!(dec.as_slice().len(), 16);

        let bands: Vec<_> = dec.bands().collect();
        assert_eq!(bands.len(), 4);
        assert_eq!(bands[0], dec.approx());
        assert_eq!(bands[1], dec.detail(3));
        assert_eq!(bands[2], dec.detail(2));
        assert_eq!(bands[3], dec.detail(1));

        let mut mutable = dec.clone();
        mutable.as_mut_slice()[0] = 42.0;
        assert_eq!(mutable.approx()[0], 42.0);
    }
}
