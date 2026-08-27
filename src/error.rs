use std::error::Error;
use std::fmt::{self, Display, Formatter};

/// An error produced while defining, planning, or applying a transform.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum WaveletError {
    /// A transform cannot be planned for an empty signal.
    EmptySignal,
    /// The selected boundary mode is undefined for this signal length.
    BoundaryRequiresLongerSignal {
        /// The requested signal length.
        len: usize,
        /// The minimum supported length.
        minimum: usize,
        /// The boundary mode's stable name.
        boundary: &'static str,
    },
    /// A custom filter bank is structurally invalid.
    InvalidFilterBank(&'static str),
    /// A boundary mode name is not recognized.
    UnknownBoundary {
        /// The unrecognized name.
        name: String,
    },
    /// A wavelet name is not recognized.
    UnknownWavelet {
        /// The unrecognized name.
        name: String,
    },
    /// The requested built-in wavelet is not available yet.
    UnsupportedWavelet {
        /// The wavelet family name.
        family: &'static str,
        /// The requested order encoded for that family.
        order: String,
    },
    /// A requested decomposition level exceeds the boundary-safe maximum.
    InvalidLevel {
        /// The requested number of levels.
        requested: usize,
        /// The maximum supported number of levels.
        maximum: usize,
    },
    /// Approximation and detail bands have different lengths.
    CoefficientLengthMismatch {
        /// The approximation-band length.
        approx: usize,
        /// The detail-band length.
        detail: usize,
    },
    /// A coefficient length cannot describe an inverse transform for the
    /// selected filter and boundary mode.
    InvalidCoefficientLength {
        /// The length of each coefficient band.
        len: usize,
        /// The reconstruction filter length.
        filter_len: usize,
        /// The boundary mode's stable name.
        boundary: &'static str,
    },
}

impl Display for WaveletError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySignal => f.write_str("a wavelet transform requires a non-empty signal"),
            Self::BoundaryRequiresLongerSignal {
                len,
                minimum,
                boundary,
            } => write!(
                f,
                "boundary mode {boundary:?} requires a signal of length at least {minimum}, got {len}"
            ),
            Self::InvalidFilterBank(reason) => write!(f, "invalid filter bank: {reason}"),
            Self::UnknownBoundary { name } => write!(f, "unknown boundary mode {name:?}"),
            Self::UnknownWavelet { name } => write!(f, "unknown wavelet {name:?}"),
            Self::UnsupportedWavelet { family, order } => {
                write!(f, "unsupported {family} wavelet order {order}")
            }
            Self::InvalidLevel { requested, maximum } => write!(
                f,
                "decomposition level {requested} exceeds the boundary-safe maximum {maximum}"
            ),
            Self::CoefficientLengthMismatch { approx, detail } => write!(
                f,
                "approximation and detail lengths differ: {approx} != {detail}"
            ),
            Self::InvalidCoefficientLength {
                len,
                filter_len,
                boundary,
            } => write!(
                f,
                "coefficient length {len} is invalid for filter length {filter_len} and boundary mode {boundary:?}"
            ),
        }
    }
}

impl Error for WaveletError {}
