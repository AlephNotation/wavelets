mod butterfly;
mod layout;
mod plan;

pub use self::layout::Decomposition;
pub use self::plan::WavedecPlan;

use crate::WaveletError;

/// Selects the number of levels in a multilevel decomposition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Level {
    /// Use the largest level at which at least one coefficient is unaffected by
    /// boundary extension.
    Max,
    /// Use exactly this many levels, rejecting values above [`Level::Max`].
    Exact(usize),
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

#[cfg(test)]
mod tests;
