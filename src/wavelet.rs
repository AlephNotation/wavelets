use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::WaveletError;
use crate::coefficients;

static NEXT_WAVELET_ID: AtomicU64 = AtomicU64::new(1);

/// The family to which a wavelet belongs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum WaveletFamily {
    /// Haar, equivalently Daubechies order 1.
    Haar,
    /// Daubechies extremal-phase orthogonal wavelets.
    Daubechies,
    /// Least-asymmetric orthogonal wavelets.
    Symlet,
    /// Coiflets.
    Coiflet,
    /// Biorthogonal wavelets.
    Biorthogonal,
    /// Reverse-biorthogonal wavelets.
    ReverseBiorthogonal,
    /// A caller-supplied filter bank.
    Custom,
}

#[derive(Debug)]
struct WaveletData {
    id: u64,
    name: Arc<str>,
    family: WaveletFamily,
    dec_lo: Box<[f64]>,
    dec_hi: Box<[f64]>,
    rec_lo: Box<[f64]>,
    rec_hi: Box<[f64]>,
    vanishing_moments: Option<usize>,
    orthogonal: bool,
    biorthogonal: bool,
}

/// An immutable discrete-wavelet filter bank with shared ownership.
///
/// Cloning a `Wavelet` only clones an [`Arc`]. Filter coefficients use the
/// same ordering as PyWavelets' `filter_bank` property.
#[derive(Clone, Debug)]
pub struct Wavelet(Arc<WaveletData>);

impl Wavelet {
    /// Constructs the Haar wavelet (`db1`).
    pub fn haar() -> Self {
        Self::orthogonal_from_low_pass(
            "haar",
            WaveletFamily::Haar,
            1,
            coefficients::daubechies(1).expect("generated db1 coefficients are present"),
        )
    }

    /// Constructs a built-in wavelet from its canonical PyWavelets name.
    ///
    /// Names for families that are recognized but not implemented return
    /// [`WaveletError::UnsupportedWavelet`].
    pub fn from_name(name: &str) -> Result<Self, WaveletError> {
        if name == "haar" {
            return Ok(Self::haar());
        }
        if let Some(order) = parse_integer_name(name, "db") {
            return Self::daubechies(order);
        }
        if let Some(order) = parse_integer_name(name, "sym") {
            return Self::symlet(order);
        }
        if let Some(order) = parse_integer_name(name, "coif") {
            return Self::coiflet(order);
        }
        if let Some((reconstruction, decomposition)) = parse_pair_name(name, "bior") {
            return Self::biorthogonal(reconstruction, decomposition);
        }
        if let Some((reconstruction, decomposition)) = parse_pair_name(name, "rbio") {
            return Self::reverse_biorthogonal(reconstruction, decomposition);
        }
        Err(WaveletError::UnknownWavelet {
            name: name.to_owned(),
        })
    }

    /// Constructs a Daubechies wavelet.
    ///
    /// Orders `db1` through `db38` are available.
    pub fn daubechies(n: usize) -> Result<Self, WaveletError> {
        let Some(dec_lo) = coefficients::daubechies(n) else {
            return Err(WaveletError::UnsupportedWavelet {
                family: "Daubechies",
                order: n.to_string(),
            });
        };
        Ok(Self::orthogonal_from_low_pass(
            &format!("db{n}"),
            WaveletFamily::Daubechies,
            n,
            dec_lo,
        ))
    }

    /// Constructs a least-asymmetric Symlet wavelet.
    ///
    /// Orders `sym2` through `sym20` are available.
    pub fn symlet(n: usize) -> Result<Self, WaveletError> {
        let Some(dec_lo) = coefficients::symlet(n) else {
            return Err(WaveletError::UnsupportedWavelet {
                family: "Symlet",
                order: n.to_string(),
            });
        };
        Ok(Self::orthogonal_from_low_pass(
            &format!("sym{n}"),
            WaveletFamily::Symlet,
            n,
            dec_lo,
        ))
    }

    /// Constructs a Coiflet wavelet.
    pub fn coiflet(n: usize) -> Result<Self, WaveletError> {
        Err(WaveletError::UnsupportedWavelet {
            family: "Coiflet",
            order: n.to_string(),
        })
    }

    /// Constructs a biorthogonal wavelet identified by reconstruction and
    /// decomposition orders.
    pub fn biorthogonal(nr: usize, nd: usize) -> Result<Self, WaveletError> {
        Err(WaveletError::UnsupportedWavelet {
            family: "biorthogonal",
            order: format!("{nr}.{nd}"),
        })
    }

    /// Constructs a reverse-biorthogonal wavelet identified by reconstruction
    /// and decomposition orders.
    pub fn reverse_biorthogonal(nr: usize, nd: usize) -> Result<Self, WaveletError> {
        Err(WaveletError::UnsupportedWavelet {
            family: "reverse biorthogonal",
            order: format!("{nr}.{nd}"),
        })
    }

    /// Constructs a custom filter bank.
    ///
    /// All four filters must have the same, positive, even length and contain
    /// only finite values. Mathematical perfect reconstruction is a property of
    /// the supplied filters and is not inferred from approximate coefficients.
    pub fn from_filters(
        dec_lo: &[f64],
        dec_hi: &[f64],
        rec_lo: &[f64],
        rec_hi: &[f64],
    ) -> Result<Self, WaveletError> {
        let len = dec_lo.len();
        if len < 2 {
            return Err(WaveletError::InvalidFilterBank(
                "filters must contain at least two coefficients",
            ));
        }
        if !len.is_multiple_of(2) {
            return Err(WaveletError::InvalidFilterBank(
                "filter length must be even",
            ));
        }
        if [dec_hi.len(), rec_lo.len(), rec_hi.len()]
            .into_iter()
            .any(|candidate| candidate != len)
        {
            return Err(WaveletError::InvalidFilterBank(
                "all four filters must have the same length",
            ));
        }
        if dec_lo
            .iter()
            .chain(dec_hi)
            .chain(rec_lo)
            .chain(rec_hi)
            .any(|coefficient| !coefficient.is_finite())
        {
            return Err(WaveletError::InvalidFilterBank(
                "filter coefficients must be finite",
            ));
        }

        Ok(Self::new(
            "custom",
            WaveletFamily::Custom,
            dec_lo,
            dec_hi,
            rec_lo,
            rec_hi,
            None,
            false,
            false,
        ))
    }

    fn orthogonal_from_low_pass(
        name: &str,
        family: WaveletFamily,
        vanishing_moments: usize,
        dec_lo: &[f64],
    ) -> Self {
        let len = dec_lo.len();
        let dec_hi: Vec<_> = dec_lo
            .iter()
            .rev()
            .enumerate()
            .map(|(index, coefficient)| {
                if index % 2 == 0 {
                    -coefficient
                } else {
                    *coefficient
                }
            })
            .collect();
        let rec_lo: Vec<_> = dec_lo.iter().rev().copied().collect();
        let rec_hi: Vec<_> = dec_hi.iter().rev().copied().collect();
        debug_assert_eq!(len, dec_hi.len());
        Self::new(
            name,
            family,
            dec_lo,
            &dec_hi,
            &rec_lo,
            &rec_hi,
            Some(vanishing_moments),
            true,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        name: &str,
        family: WaveletFamily,
        dec_lo: &[f64],
        dec_hi: &[f64],
        rec_lo: &[f64],
        rec_hi: &[f64],
        vanishing_moments: Option<usize>,
        orthogonal: bool,
        biorthogonal: bool,
    ) -> Self {
        Self(Arc::new(WaveletData {
            id: NEXT_WAVELET_ID.fetch_add(1, Ordering::Relaxed),
            name: Arc::from(name),
            family,
            dec_lo: dec_lo.into(),
            dec_hi: dec_hi.into(),
            rec_lo: rec_lo.into(),
            rec_hi: rec_hi.into(),
            vanishing_moments,
            orthogonal,
            biorthogonal,
        }))
    }

    /// Returns the conventional short name.
    pub fn name(&self) -> &str {
        &self.0.name
    }

    /// Returns the wavelet family.
    pub fn family(&self) -> WaveletFamily {
        self.0.family
    }

    /// Returns the decomposition low-pass filter.
    pub fn dec_lo(&self) -> &[f64] {
        &self.0.dec_lo
    }

    /// Returns the decomposition high-pass filter.
    pub fn dec_hi(&self) -> &[f64] {
        &self.0.dec_hi
    }

    /// Returns the reconstruction low-pass filter.
    pub fn rec_lo(&self) -> &[f64] {
        &self.0.rec_lo
    }

    /// Returns the reconstruction high-pass filter.
    pub fn rec_hi(&self) -> &[f64] {
        &self.0.rec_hi
    }

    /// Returns the common filter length.
    pub fn filter_len(&self) -> usize {
        self.0.dec_lo.len()
    }

    /// Returns the number of vanishing wavelet moments when known.
    pub fn vanishing_moments(&self) -> Option<usize> {
        self.0.vanishing_moments
    }

    /// Reports whether this wavelet is orthogonal.
    pub fn is_orthogonal(&self) -> bool {
        self.0.orthogonal
    }

    /// Reports whether this wavelet is biorthogonal.
    pub fn is_biorthogonal(&self) -> bool {
        self.0.biorthogonal
    }

    pub(crate) fn id(&self) -> u64 {
        self.0.id
    }

    pub(crate) fn has_same_filter_bank(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
            || (self.dec_lo() == other.dec_lo()
                && self.dec_hi() == other.dec_hi()
                && self.rec_lo() == other.rec_lo()
                && self.rec_hi() == other.rec_hi())
    }
}

impl FromStr for Wavelet {
    type Err = WaveletError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::from_name(name)
    }
}

fn parse_integer_name(name: &str, prefix: &str) -> Option<usize> {
    let digits = name.strip_prefix(prefix)?;
    if digits.is_empty() || (digits.len() > 1 && digits.starts_with('0')) {
        return None;
    }
    digits.parse().ok()
}

fn parse_pair_name(name: &str, prefix: &str) -> Option<(usize, usize)> {
    let pair = name.strip_prefix(prefix)?;
    let (first, second) = pair.split_once('.')?;
    if first.is_empty()
        || second.is_empty()
        || (first.len() > 1 && first.starts_with('0'))
        || (second.len() > 1 && second.starts_with('0'))
    {
        return None;
    }
    Some((first.parse().ok()?, second.parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn db2_is_generated_in_pywavelets_order() {
        let wavelet = Wavelet::daubechies(2).unwrap();
        let expected = [
            -0.129_409_522_551_260_37,
            0.224_143_868_042_013_4,
            0.836_516_303_737_807_9,
            0.482_962_913_144_534_16,
        ];
        for (actual, expected) in wavelet.dec_lo().iter().zip(expected) {
            assert!((actual - expected).abs() <= f64::EPSILON);
        }
    }

    #[test]
    fn canonical_names_construct_built_ins() {
        assert_eq!(Wavelet::from_name("haar").unwrap().name(), "haar");
        assert_eq!("db4".parse::<Wavelet>().unwrap().name(), "db4");
        assert_eq!(Wavelet::from_name("sym4").unwrap().name(), "sym4");
        assert_eq!(
            Wavelet::from_name("db04").unwrap_err(),
            WaveletError::UnknownWavelet {
                name: "db04".to_owned()
            }
        );
    }

    #[test]
    fn all_daubechies_orders_are_available() {
        for order in 1..=38 {
            let wavelet = Wavelet::daubechies(order).unwrap();
            assert_eq!(wavelet.name(), format!("db{order}"));
            assert_eq!(wavelet.filter_len(), 2 * order);
            assert_eq!(wavelet.vanishing_moments(), Some(order));
        }
        assert!(Wavelet::daubechies(0).is_err());
        assert!(Wavelet::daubechies(39).is_err());
    }

    #[test]
    fn all_symlet_orders_are_available() {
        for order in 2..=20 {
            let wavelet = Wavelet::symlet(order).unwrap();
            assert_eq!(wavelet.name(), format!("sym{order}"));
            assert_eq!(wavelet.family(), WaveletFamily::Symlet);
            assert_eq!(wavelet.filter_len(), 2 * order);
            assert_eq!(wavelet.vanishing_moments(), Some(order));
            assert!(wavelet.is_orthogonal());
        }
        assert!(Wavelet::symlet(1).is_err());
        assert!(Wavelet::symlet(21).is_err());
    }

    #[test]
    fn custom_filters_require_a_finite_even_common_length() {
        assert!(Wavelet::from_filters(&[1.0], &[1.0], &[1.0], &[1.0]).is_err());
        assert!(
            Wavelet::from_filters(
                &[1.0, 2.0, 3.0],
                &[1.0, 2.0, 3.0],
                &[1.0, 2.0, 3.0],
                &[1.0, 2.0, 3.0]
            )
            .is_err()
        );
        assert!(
            Wavelet::from_filters(&[1.0, f64::NAN], &[1.0, 2.0], &[1.0, 2.0], &[1.0, 2.0]).is_err()
        );
    }
}
