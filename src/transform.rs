use fearless_simd::Level as SimdLevel;

use crate::plan::create_dwt_plan;
use crate::{Boundary, Dwt, Wavelet, WaveletError, WaveletNum};

/// Computes a single-level one-dimensional discrete wavelet transform.
///
/// This allocating convenience function mirrors PyWavelets' `dwt` operation.
/// Use [`crate::DwtPlanner`] when repeatedly transforming a fixed signal length.
pub fn dwt<T: WaveletNum>(
    signal: &[T],
    wavelet: &Wavelet,
    boundary: Boundary,
) -> Result<(Vec<T>, Vec<T>), WaveletError> {
    let plan = create_dwt_plan(signal.len(), wavelet, boundary, SimdLevel::new())?;
    Ok(plan.forward(signal))
}

/// Reconstructs a signal from one approximation and one detail band.
///
/// Like standalone PyWavelets `idwt`, this function reconstructs the canonical
/// even signal length implied by the coefficient and filter lengths. Therefore
/// `idwt(dwt(odd_length_signal))` contains one additional boundary-derived
/// sample. A fixed-length plan remembers and reconstructs the exact original
/// length instead.
pub fn idwt<T: WaveletNum>(
    approx: &[T],
    detail: &[T],
    wavelet: &Wavelet,
    boundary: Boundary,
) -> Result<Vec<T>, WaveletError> {
    if approx.len() != detail.len() {
        return Err(WaveletError::CoefficientLengthMismatch {
            approx: approx.len(),
            detail: detail.len(),
        });
    }
    let signal_len = inverse_signal_len(approx.len(), wavelet.filter_len(), boundary).ok_or(
        WaveletError::InvalidCoefficientLength {
            len: approx.len(),
            filter_len: wavelet.filter_len(),
            boundary: boundary.as_str(),
        },
    )?;
    let plan = create_dwt_plan(signal_len, wavelet, boundary, SimdLevel::new())?;
    debug_assert_eq!(plan.coeff_len(), approx.len());
    Ok(plan.inverse(approx, detail))
}

fn inverse_signal_len(
    coefficient_len: usize,
    filter_len: usize,
    boundary: Boundary,
) -> Option<usize> {
    if coefficient_len == 0 {
        return None;
    }
    let doubled = coefficient_len.checked_mul(2)?;
    match boundary {
        Boundary::Periodization => Some(doubled),
        _ => doubled.checked_add(2)?.checked_sub(filter_len),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_length_inverse_matches_pywavelets_standalone_semantics() {
        let signal = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let wavelet = Wavelet::daubechies(2).unwrap();

        for (boundary, expected_extra) in [
            (Boundary::Zero, 0.0),
            (Boundary::Constant, 5.0),
            (Boundary::Symmetric, 5.0),
            (Boundary::Reflect, 4.0),
            (Boundary::Periodic, 1.0),
            (Boundary::Smooth, 6.0),
            (Boundary::Antisymmetric, -5.0),
            (Boundary::Antireflect, 6.0),
            (Boundary::Periodization, 5.0),
        ] {
            let (approx, detail) = dwt(&signal, &wavelet, boundary).unwrap();
            let reconstructed = idwt(&approx, &detail, &wavelet, boundary).unwrap();
            assert_eq!(reconstructed.len(), 6);
            for (actual, expected) in reconstructed.iter().zip(signal) {
                assert!((actual - expected).abs() < 1e-12, "{boundary}");
            }
            assert!(
                (reconstructed[5] - expected_extra).abs() < 1e-12,
                "{boundary}: {} != {expected_extra}",
                reconstructed[5]
            );
        }
    }

    #[test]
    fn inverse_rejects_structurally_invalid_bands() {
        let wavelet = Wavelet::daubechies(4).unwrap();
        assert!(matches!(
            idwt(&[1.0], &[1.0, 2.0], &wavelet, Boundary::Symmetric),
            Err(WaveletError::CoefficientLengthMismatch { .. })
        ));
        assert!(matches!(
            idwt::<f64>(&[], &[], &wavelet, Boundary::Symmetric),
            Err(WaveletError::InvalidCoefficientLength { .. })
        ));
    }
}
