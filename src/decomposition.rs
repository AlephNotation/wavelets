use std::ops::Range;

use crate::{Boundary, DwtPlanner, Wavelet, WaveletError, WaveletNum};

/// Selects the number of levels in a multilevel decomposition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Level {
    /// Use the largest level at which at least one coefficient is unaffected by
    /// boundary extension.
    Max,
    /// Use exactly this many levels, rejecting values above [`Level::Max`].
    Exact(usize),
}

/// An owned multilevel decomposition stored in one contiguous allocation.
///
/// The physical layout is `cA_L, cD_L, ..., cD_1`, while [`Self::detail`]
/// addresses detail bands by their natural one-based level.
#[derive(Clone, Debug)]
pub struct Decomposition<T> {
    buffer: Vec<T>,
    approx: Range<usize>,
    details: Vec<Range<usize>>,
    wavelet: Wavelet,
    boundary: Boundary,
    input_lengths: Vec<usize>,
}

impl<T> Decomposition<T> {
    /// Returns the number of decomposition levels.
    pub fn levels(&self) -> usize {
        self.details.len()
    }

    /// Returns the coarsest approximation band, `cA_L`.
    pub fn approx(&self) -> &[T] {
        &self.buffer[self.approx.clone()]
    }

    /// Returns `cD_level` for a one-based level in `1..=levels()`.
    ///
    /// # Panics
    ///
    /// Panics when `level` is zero or greater than [`Self::levels`].
    pub fn detail(&self, level: usize) -> &[T] {
        let range = self
            .details
            .get(level.checked_sub(1).expect("detail levels are one-based"))
            .expect("detail level is out of range")
            .clone();
        &self.buffer[range]
    }

    /// Mutably returns the coarsest approximation band, `cA_L`.
    pub fn approx_mut(&mut self) -> &mut [T] {
        &mut self.buffer[self.approx.clone()]
    }

    /// Mutably returns `cD_level` for a one-based level in `1..=levels()`.
    ///
    /// # Panics
    ///
    /// Panics when `level` is zero or greater than [`Self::levels`].
    pub fn detail_mut(&mut self, level: usize) -> &mut [T] {
        let range = self
            .details
            .get(level.checked_sub(1).expect("detail levels are one-based"))
            .expect("detail level is out of range")
            .clone();
        &mut self.buffer[range]
    }

    /// Returns the original signal length.
    pub fn original_len(&self) -> usize {
        self.input_lengths.first().copied().unwrap_or(0)
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

/// Computes a multilevel one-dimensional wavelet decomposition.
pub fn wavedec<T: WaveletNum>(
    signal: &[T],
    wavelet: &Wavelet,
    boundary: Boundary,
    level: Level,
) -> Result<Decomposition<T>, WaveletError> {
    if signal.is_empty() {
        return Err(WaveletError::EmptySignal);
    }
    let maximum = dwt_max_level(signal.len(), wavelet.filter_len());
    let levels = match level {
        Level::Max => maximum,
        Level::Exact(requested) if requested <= maximum => requested,
        Level::Exact(requested) => {
            return Err(WaveletError::InvalidLevel { requested, maximum });
        }
    };

    let mut planner = DwtPlanner::<T>::new();
    let mut current = signal.to_vec();
    let mut details = Vec::with_capacity(levels);
    let mut input_lengths = Vec::with_capacity(levels.max(1));
    input_lengths.push(signal.len());

    for decomposition_level in 0..levels {
        let plan = planner.plan_dwt(current.len(), wavelet, boundary)?;
        let (next, detail) = plan.forward(&current);
        details.push(detail);
        current = next;
        if decomposition_level + 1 < levels {
            input_lengths.push(current.len());
        }
    }

    let total_len = current.len() + details.iter().map(Vec::len).sum::<usize>();
    let mut buffer = Vec::with_capacity(total_len);
    buffer.extend_from_slice(&current);
    let approx = 0..current.len();
    let mut detail_ranges = vec![0..0; levels];
    for (zero_based_level, detail) in details.into_iter().enumerate().rev() {
        let start = buffer.len();
        buffer.extend(detail);
        detail_ranges[zero_based_level] = start..buffer.len();
    }

    Ok(Decomposition {
        buffer,
        approx,
        details: detail_ranges,
        wavelet: wavelet.clone(),
        boundary,
        input_lengths,
    })
}

/// Reconstructs a signal from a decomposition created by [`wavedec`].
pub fn waverec<T: WaveletNum>(dec: &Decomposition<T>) -> Result<Vec<T>, WaveletError> {
    let mut current = dec.approx().to_vec();
    let mut planner = DwtPlanner::<T>::new();
    for level in (1..=dec.levels()).rev() {
        let signal_len = dec.input_lengths[level - 1];
        let plan = planner.plan_dwt(signal_len, &dec.wavelet, dec.boundary)?;
        current = plan.inverse(&current, dec.detail(level));
    }
    Ok(current)
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
    }
}
