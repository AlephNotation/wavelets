#![deny(unsafe_code)]

use wavelets::{Boundary, Wavelet};

pub const MAX_SIGNAL_LEN: usize = 1024;
pub const MAX_FILTER_LEN: usize = 80;

#[derive(Debug)]
pub struct TransformCase {
    pub wavelet: Wavelet,
    pub boundary: Boundary,
    pub use_f32: bool,
    pub samples: Vec<f64>,
}

impl TransformCase {
    pub fn decode(data: &[u8], header_len: usize) -> Self {
        let order_selector = data.first().copied().unwrap_or(0);
        let mode_selector = data.get(1).copied().unwrap_or(0);
        Self {
            wavelet: built_in_wavelet(order_selector),
            boundary: boundary(mode_selector),
            use_f32: mode_selector & 0x80 != 0,
            samples: decode_samples(data.get(header_len..).unwrap_or_default()),
        }
    }
}

fn built_in_wavelet(selector: u8) -> Wavelet {
    const DAUBECHIES_COUNT: usize = 38;
    const SYMLET_COUNT: usize = 19;
    const COIFLET_COUNT: usize = 17;
    const BIORTHOGONAL_ORDERS: [(usize, usize); 15] = [
        (1, 1),
        (1, 3),
        (1, 5),
        (2, 2),
        (2, 4),
        (2, 6),
        (2, 8),
        (3, 1),
        (3, 3),
        (3, 5),
        (3, 7),
        (3, 9),
        (4, 4),
        (5, 5),
        (6, 8),
    ];

    let orthogonal_count = DAUBECHIES_COUNT + SYMLET_COUNT + COIFLET_COUNT;
    let biorthogonal_count = BIORTHOGONAL_ORDERS.len();
    let index = usize::from(selector) % (orthogonal_count + 2 * biorthogonal_count);
    if index < DAUBECHIES_COUNT {
        Wavelet::daubechies(index + 1).expect("normalized Daubechies order is supported")
    } else if index < DAUBECHIES_COUNT + SYMLET_COUNT {
        Wavelet::symlet(index - DAUBECHIES_COUNT + 2).expect("normalized Symlet order is supported")
    } else if index < orthogonal_count {
        Wavelet::coiflet(index - DAUBECHIES_COUNT - SYMLET_COUNT + 1)
            .expect("normalized Coiflet order is supported")
    } else {
        let family_index = index - orthogonal_count;
        let reverse = family_index >= biorthogonal_count;
        let pair = BIORTHOGONAL_ORDERS[family_index % biorthogonal_count];
        if reverse {
            Wavelet::reverse_biorthogonal(pair.0, pair.1)
                .expect("normalized reverse-biorthogonal pair is supported")
        } else {
            Wavelet::biorthogonal(pair.0, pair.1)
                .expect("normalized biorthogonal pair is supported")
        }
    }
}

pub fn boundary(selector: u8) -> Boundary {
    match selector % 9 {
        0 => Boundary::Zero,
        1 => Boundary::Constant,
        2 => Boundary::Symmetric,
        3 => Boundary::Reflect,
        4 => Boundary::Periodic,
        5 => Boundary::Smooth,
        6 => Boundary::Antisymmetric,
        7 => Boundary::Antireflect,
        8 => Boundary::Periodization,
        _ => unreachable!("a remainder modulo nine is in range"),
    }
}

pub fn decode_samples(data: &[u8]) -> Vec<f64> {
    data.chunks(2)
        .take(MAX_SIGNAL_LEN)
        .map(|chunk| {
            let bytes = [chunk[0], chunk.get(1).copied().unwrap_or(0)];
            f64::from(i16::from_le_bytes(bytes)) / 32768.0
        })
        .collect()
}

pub fn assert_reconstruction_f64(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    let scale = expected.iter().copied().map(f64::abs).fold(1.0, f64::max);
    let tolerance = 1.0e-9 * scale;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite(),
            "non-finite f64 reconstruction at {index}"
        );
        assert!(
            (actual - expected).abs() <= tolerance,
            "f64 reconstruction mismatch at {index}: {actual:e} != {expected:e} (tol {tolerance:e})"
        );
    }
}

pub fn assert_reconstruction_f32(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let scale = expected.iter().copied().map(f32::abs).fold(1.0, f32::max);
    let tolerance = 2.0e-3 * scale;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite(),
            "non-finite f32 reconstruction at {index}"
        );
        assert!(
            (actual - expected).abs() <= tolerance,
            "f32 reconstruction mismatch at {index}: {actual:e} != {expected:e} (tol {tolerance:e})"
        );
    }
}

pub fn assert_same_f64(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

pub fn assert_same_f32(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

pub fn decode_filter_coefficients(data: &[u8]) -> Vec<f64> {
    data.as_chunks::<8>()
        .0
        .iter()
        .take(MAX_FILTER_LEN)
        .map(|chunk| {
            let bits = u64::from_le_bytes(*chunk);
            let value = f64::from_bits(bits);
            if value.is_finite() {
                value
            } else {
                f64::from_bits(bits & 0x7fef_ffff_ffff_ffff)
            }
        })
        .collect()
}

pub fn make_even_nonempty(values: &mut Vec<f64>) {
    if values.len() < 2 {
        values.resize(2, 0.0);
    }
    if !values.len().is_multiple_of(2) {
        values.push(0.0);
    }
}
