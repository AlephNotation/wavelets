use std::ops::Range;
use std::sync::Arc;

use crate::{Boundary, Wavelet};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct DecompositionLayout {
    pub(super) approx: Range<usize>,
    pub(super) details: Box<[Range<usize>]>,
    pub(super) input_lengths: Box<[usize]>,
    pub(super) buffer_len: usize,
}

impl DecompositionLayout {
    pub(super) fn new(input_lengths: Vec<usize>, coefficient_lengths: &[usize]) -> Self {
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
/// allocated by [`super::WavedecPlan::allocate_decomposition`] can be overwritten and
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
    pub(super) buffer: Vec<T>,
    pub(super) layout: Arc<DecompositionLayout>,
    pub(super) wavelet: Wavelet,
    pub(super) boundary: Boundary,
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
