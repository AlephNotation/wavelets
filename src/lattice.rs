#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
use std::mem::size_of;

#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
use crate::{Wavelet, WaveletNum, lattice_coefficients};

#[derive(Clone, Copy, Debug)]
pub(crate) struct LatticeSection<T> {
    pub(crate) q: T,
    pub(crate) chart: u8,
    pub(crate) determinant: i8,
}

#[derive(Debug)]
pub(crate) struct LatticeFilter<T> {
    pub(crate) sections: Box<[LatticeSection<T>]>,
    pub(crate) scale: T,
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
impl<T: WaveletNum> LatticeFilter<T> {
    pub(crate) fn new(wavelet: &Wavelet) -> Option<Self> {
        // The generated factors currently carry an f64 error analysis. In
        // particular, coif17's unnormalized cascade has too much intermediate
        // growth to opt f32 into the same representation without separate
        // evidence.
        if size_of::<T>() != size_of::<f64>() || !wavelet.is_orthogonal() {
            return None;
        }
        let factors = lattice_coefficients::analysis(wavelet.dec_lo())?;
        Some(Self {
            sections: factors
                .sections
                .iter()
                .map(|section| LatticeSection {
                    q: T::from_f64(section.q),
                    chart: section.chart,
                    determinant: section.determinant,
                })
                .collect(),
            scale: T::from_f64(factors.scale),
        })
    }
}

#[cfg(all(
    test,
    any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")
))]
mod tests {
    use super::{LatticeFilter, LatticeSection};
    use crate::Wavelet;

    fn apply(section: LatticeSection<f64>, first: f64, second: f64) -> (f64, f64) {
        let determinant = f64::from(section.determinant);
        if section.chart == 0 {
            (
                first - determinant * section.q * second,
                section.q * first + determinant * second,
            )
        } else {
            (
                section.q * first - determinant * second,
                first + determinant * section.q * second,
            )
        }
    }

    fn impulse_response(filter: &LatticeFilter<f64>, input_channel: usize) -> Vec<[f64; 2]> {
        let mut state = vec![0.0; filter.sections.len() - 1];
        (0..filter.sections.len())
            .map(|index| {
                let mut first = f64::from(input_channel == 0 && index == 0);
                let mut second = f64::from(input_channel == 1 && index == 0);
                for (section_index, &section) in filter.sections.iter().enumerate() {
                    if section_index != 0 {
                        std::mem::swap(&mut second, &mut state[section_index - 1]);
                    }
                    (first, second) = apply(section, first, second);
                }
                [filter.scale * first, filter.scale * second]
            })
            .collect()
    }

    #[test]
    fn generated_factors_reconstruct_crate_polyphase_banks() {
        for wavelet in [
            Wavelet::daubechies(20).unwrap(),
            Wavelet::symlet(20).unwrap(),
            Wavelet::daubechies(38).unwrap(),
            Wavelet::coiflet(17).unwrap(),
        ] {
            let filter = LatticeFilter::<f64>::new(&wavelet).unwrap();
            let first_channel = impulse_response(&filter, 0);
            let second_channel = impulse_response(&filter, 1);
            for tap_pair in 0..filter.sections.len() {
                let expected = [
                    [
                        wavelet.dec_lo()[2 * tap_pair + 1],
                        wavelet.dec_lo()[2 * tap_pair],
                    ],
                    [
                        wavelet.dec_hi()[2 * tap_pair + 1],
                        wavelet.dec_hi()[2 * tap_pair],
                    ],
                ];
                let actual = [first_channel[tap_pair], second_channel[tap_pair]];
                for output in 0..2 {
                    for input in 0..2 {
                        let error = (actual[input][output] - expected[output][input]).abs();
                        assert!(
                            error <= 2.0e-15,
                            "{} polyphase[{tap_pair}][{output}][{input}] error {error:.3e}",
                            wavelet.name()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn factors_are_f64_only_until_f32_growth_is_bounded() {
        assert!(LatticeFilter::<f64>::new(&Wavelet::daubechies(38).unwrap()).is_some());
        assert!(LatticeFilter::<f32>::new(&Wavelet::daubechies(38).unwrap()).is_none());
    }
}
