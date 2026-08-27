use std::fmt::{self, Display, Formatter};
use std::str::FromStr;

use crate::WaveletError;

/// Signal extension applied beyond the original signal boundaries.
///
/// Names and semantics match PyWavelets. [`Boundary::Symmetric`] is the
/// conventional default, but callers select a mode explicitly when planning.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum Boundary {
    /// Extend with zeroes.
    Zero,
    /// Repeat the nearest edge sample.
    Constant,
    /// Half-sample symmetric extension (edge samples are repeated).
    #[default]
    Symmetric,
    /// Whole-sample symmetric extension (edge samples are not repeated).
    Reflect,
    /// Periodic extension without coefficient shortening.
    Periodic,
    /// First-derivative extrapolation from each edge.
    Smooth,
    /// Half-sample antisymmetric extension.
    Antisymmetric,
    /// Whole-sample antisymmetric reflection about each edge sample.
    Antireflect,
    /// Periodic extension with the smallest possible coefficient count.
    Periodization,
}

impl Boundary {
    /// Returns the canonical PyWavelets mode name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Zero => "zero",
            Self::Constant => "constant",
            Self::Symmetric => "symmetric",
            Self::Reflect => "reflect",
            Self::Periodic => "periodic",
            Self::Smooth => "smooth",
            Self::Antisymmetric => "antisymmetric",
            Self::Antireflect => "antireflect",
            Self::Periodization => "periodization",
        }
    }
}

impl Display for Boundary {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Boundary {
    type Err = WaveletError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "zero" => Ok(Self::Zero),
            "constant" => Ok(Self::Constant),
            "symmetric" => Ok(Self::Symmetric),
            "reflect" => Ok(Self::Reflect),
            "periodic" => Ok(Self::Periodic),
            "smooth" => Ok(Self::Smooth),
            "antisymmetric" => Ok(Self::Antisymmetric),
            "antireflect" => Ok(Self::Antireflect),
            "periodization" => Ok(Self::Periodization),
            _ => Err(WaveletError::UnknownBoundary {
                name: name.to_owned(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_names_round_trip() {
        for boundary in [
            Boundary::Zero,
            Boundary::Constant,
            Boundary::Symmetric,
            Boundary::Reflect,
            Boundary::Periodic,
            Boundary::Smooth,
            Boundary::Antisymmetric,
            Boundary::Antireflect,
            Boundary::Periodization,
        ] {
            assert_eq!(boundary.as_str().parse(), Ok(boundary));
            assert_eq!(boundary.to_string(), boundary.as_str());
        }
    }

    #[test]
    fn unknown_name_is_rejected() {
        assert_eq!(
            "mirror".parse::<Boundary>(),
            Err(WaveletError::UnknownBoundary {
                name: "mirror".to_owned()
            })
        );
    }
}
