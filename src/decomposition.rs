use std::ops::Range;
use std::sync::Arc;

use fearless_simd::Level as SimdLevel;

use crate::plan::{PlannedDwt, PreparedFilterBank, validate_plan};
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

        let coefficient_lengths: Vec<_> = level_plans.iter().map(|plan| plan.coeff_len()).collect();
        let temp_a_len = coefficient_lengths.first().copied().unwrap_or(0);
        let temp_b_len = if levels >= 3 {
            coefficient_lengths[1]
        } else {
            0
        };
        let kernel_scratch_len = level_plans
            .iter()
            .map(|plan| plan.scratch_len())
            .max()
            .unwrap_or(0);

        Ok(Self {
            wavelet: wavelet.clone(),
            boundary,
            level_plans: level_plans.into_boxed_slice(),
            layout: Arc::new(DecompositionLayout::new(
                input_lengths,
                &coefficient_lengths,
            )),
            temp_a_len: if levels >= 2 { temp_a_len } else { 0 },
            temp_b_len,
            kernel_scratch_len,
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
    pub fn forward(&self, signal: &[T]) -> Decomposition<T> {
        let mut decomposition = self.allocate_decomposition();
        let mut scratch = vec![T::zero(); self.scratch_len()];
        self.forward_into(signal, &mut decomposition, &mut scratch);
        decomposition
    }

    /// Allocates and reconstructs the original signal.
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

/// Computes the largest decomposition level with a boundary-independent
/// coefficient, matching PyWavelets' `dwt_max_level` definition.
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
    }
}
