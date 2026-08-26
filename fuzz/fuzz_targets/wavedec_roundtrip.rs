#![no_main]

use libfuzzer_sys::fuzz_target;
use wavelets::{Level, WaveletError, dwt_max_level, wavedec, waverec};
use wavelets_fuzz::{TransformCase, assert_reconstruction_f32, assert_reconstruction_f64};

fuzz_target!(|data: &[u8]| {
    let case = TransformCase::decode(data, 3);
    let maximum = dwt_max_level(case.samples.len(), case.wavelet.filter_len());
    let level_selector = data.get(2).copied().unwrap_or(0);
    let level = if level_selector & 0x80 != 0 {
        Level::Max
    } else {
        Level::Exact(usize::from(level_selector) % (maximum + 2))
    };

    if case.use_f32 {
        run_f32(case, level, maximum);
    } else {
        run_f64(case, level, maximum);
    }
});

fn run_f64(case: TransformCase, level: Level, maximum: usize) {
    let decomposition = wavedec(&case.samples, &case.wavelet, case.boundary, level);
    if case.samples.is_empty() {
        assert!(matches!(decomposition, Err(WaveletError::EmptySignal)));
        return;
    }
    if let Level::Exact(requested) = level
        && requested > maximum
    {
        assert!(matches!(
            decomposition,
            Err(WaveletError::InvalidLevel { .. })
        ));
        return;
    }

    let mut decomposition = decomposition.expect("normalized decomposition is valid");
    assert_layout(&decomposition, case.samples.len());
    touch_bands(&mut decomposition);
    let reconstructed = waverec(&decomposition).expect("a valid decomposition reconstructs");
    assert_reconstruction_f64(&reconstructed, &case.samples);
}

fn run_f32(case: TransformCase, level: Level, maximum: usize) {
    let samples: Vec<_> = case.samples.iter().map(|&value| value as f32).collect();
    let decomposition = wavedec(&samples, &case.wavelet, case.boundary, level);
    if samples.is_empty() {
        assert!(matches!(decomposition, Err(WaveletError::EmptySignal)));
        return;
    }
    if let Level::Exact(requested) = level
        && requested > maximum
    {
        assert!(matches!(
            decomposition,
            Err(WaveletError::InvalidLevel { .. })
        ));
        return;
    }

    let mut decomposition = decomposition.expect("normalized decomposition is valid");
    assert_layout(&decomposition, samples.len());
    touch_bands(&mut decomposition);
    let reconstructed = waverec(&decomposition).expect("a valid decomposition reconstructs");
    assert_reconstruction_f32(&reconstructed, &samples);
}

fn assert_layout<T>(decomposition: &wavelets::Decomposition<T>, original_len: usize) {
    let band_len = decomposition.approx().len()
        + (1..=decomposition.levels())
            .map(|level| decomposition.detail(level).len())
            .sum::<usize>();
    assert_eq!(band_len, decomposition.as_slice().len());
    assert_eq!(decomposition.original_len(), original_len);
}

fn touch_bands<T: Copy>(decomposition: &mut wavelets::Decomposition<T>) {
    if let Some(&value) = decomposition.approx().first() {
        decomposition.approx_mut()[0] = value;
    }
    for level in 1..=decomposition.levels() {
        if let Some(&value) = decomposition.detail(level).first() {
            decomposition.detail_mut(level)[0] = value;
        }
    }
}
