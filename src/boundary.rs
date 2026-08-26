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
    pub(crate) const fn as_str(self) -> &'static str {
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
