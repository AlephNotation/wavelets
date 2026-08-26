//! Reference values generated with PyWavelets 1.8.0.

#![allow(clippy::approx_constant)]

use std::collections::HashMap;

use serde::Deserialize;
use wavelets::{Boundary, DwtPlanner, Wavelet};

const DB2_LEN4: &[(Boundary, [f64; 3], [f64; 3])] = &[
    (
        Boundary::Zero,
        [
            -0.034_675_177_060_507_35,
            2.310_789_034_541_149,
            4.794_953_954_384_834,
        ],
        [
            -0.129_409_522_551_260_37,
            5.551_115_123_125_783e-17,
            -1.284_804_039_821_834_6,
        ],
    ),
    (
        Boundary::Constant,
        [
            1.284_804_039_821_834_8,
            2.310_789_034_541_149,
            5.173_891_336_347_846,
        ],
        [
            -0.482_962_913_144_534_1,
            5.551_115_123_125_783e-17,
            0.129_409_522_551_260_48,
        ],
    ),
    (
        Boundary::Symmetric,
        [
            1.767_766_952_966_368_9,
            2.310_789_034_541_149,
            5.303_300_858_899_107,
        ],
        [
            -0.612_372_435_695_794_5,
            5.551_115_123_125_783e-17,
            0.612_372_435_695_794_7,
        ],
    ),
    (
        Boundary::Reflect,
        [
            3.087_246_169_848_711,
            2.310_789_034_541_149,
            5.208_566_513_408_353,
        ],
        [
            -0.965_925_826_289_068_2,
            5.551_115_123_125_783e-17,
            0.258_819_045_102_521_1,
        ],
    ),
    (
        Boundary::Periodic,
        [
            4.760_278_777_324_327,
            2.310_789_034_541_149,
            4.760_278_777_324_327,
        ],
        [
            -1.414_213_562_373_095,
            5.551_115_123_125_783e-17,
            -1.414_213_562_373_095,
        ],
    ),
    (
        Boundary::Smooth,
        [
            -0.517_638_090_205_041_5,
            2.310_789_034_541_149,
            5.139_216_159_287_339,
        ],
        [0.0, 5.551_115_123_125_783e-17, -1.110_223_024_625_156_5e-16],
    ),
    (
        Boundary::Antisymmetric,
        [
            -1.837_117_307_087_383_6,
            2.310_789_034_541_149,
            4.286_607_049_870_562,
        ],
        [
            0.353_553_390_593_273_73,
            5.551_115_123_125_783e-17,
            -3.181_980_515_339_464,
        ],
    ),
    (
        Boundary::Antireflect,
        [
            -0.517_638_090_205_041_5,
            2.310_789_034_541_149,
            5.139_216_159_287_339,
        ],
        [0.0, 5.551_115_123_125_783e-17, 3.330_669_073_875_469_6e-16],
    ),
];

#[test]
fn db2_all_redundant_modes_match_pywavelets() {
    let wavelet = Wavelet::daubechies(2).unwrap();
    let signal = [1.0, 2.0, 3.0, 4.0];
    let mut planner = DwtPlanner::<f64>::new();

    for &(boundary, expected_approx, expected_detail) in DB2_LEN4 {
        let plan = planner.plan_dwt(signal.len(), &wavelet, boundary).unwrap();
        let (approx, detail) = plan.forward(&signal);
        assert_slice_close(&approx, &expected_approx, boundary);
        assert_slice_close(&detail, &expected_detail, boundary);
        assert_slice_close(&plan.inverse(&approx, &detail), &signal, boundary);
    }
}

#[test]
fn db2_periodization_matches_pywavelets() {
    let wavelet = Wavelet::daubechies(2).unwrap();
    let signal = [1.0, 2.0, 3.0, 4.0];
    let mut planner = DwtPlanner::<f64>::new();
    let plan = planner
        .plan_dwt(signal.len(), &wavelet, Boundary::Periodization)
        .unwrap();
    let (approx, detail) = plan.forward(&signal);
    assert_slice_close(
        &approx,
        &[2.828_427_124_746_190_3, 4.242_640_687_119_286],
        Boundary::Periodization,
    );
    assert_slice_close(
        &detail,
        &[-0.517_638_090_205_041_5, 1.931_851_652_578_136_4],
        Boundary::Periodization,
    );
    assert_slice_close(
        &plan.inverse(&approx, &detail),
        &signal,
        Boundary::Periodization,
    );
}

#[derive(Deserialize)]
struct Fixtures {
    generator: String,
    signals: Vec<FixtureSignal>,
    cases: Vec<FixtureCase>,
}

#[derive(Deserialize)]
struct FixtureSignal {
    len: usize,
    values: Vec<f64>,
}

#[derive(Deserialize)]
struct FixtureCase {
    wavelet: String,
    mode: String,
    len: usize,
    approx: Vec<f64>,
    detail: Vec<f64>,
}

#[test]
fn generated_fixture_matrix_matches_pywavelets() {
    let fixtures: Fixtures =
        serde_json::from_str(include_str!("fixtures/pywavelets-1.8.0.json")).unwrap();
    assert_eq!(fixtures.generator, "PyWavelets 1.8.0");
    let signals: HashMap<_, _> = fixtures
        .signals
        .into_iter()
        .map(|signal| (signal.len, signal.values))
        .collect();
    let mut planner = DwtPlanner::<f64>::new();

    for case in fixtures.cases {
        let wavelet = match case.wavelet.as_str() {
            "haar" => Wavelet::haar(),
            name => Wavelet::daubechies(
                name.strip_prefix("db")
                    .unwrap_or_else(|| panic!("unknown fixture wavelet {name}"))
                    .parse()
                    .unwrap(),
            )
            .unwrap(),
        };
        let boundary = fixture_boundary(&case.mode);
        let signal = &signals[&case.len];
        let plan = planner.plan_dwt(case.len, &wavelet, boundary).unwrap();
        let (approx, detail) = plan.forward(signal);
        let context = format!("{} {} len={}", case.wavelet, case.mode, case.len);
        let reference_tolerance = reference_tolerance(&wavelet, boundary, signal);
        assert_slice_close_with_tolerance(
            &approx,
            &case.approx,
            boundary,
            &context,
            reference_tolerance,
        );
        assert_slice_close_with_tolerance(
            &detail,
            &case.detail,
            boundary,
            &context,
            reference_tolerance,
        );
        let reconstruction_tolerance =
            1.0e-12 * signal.iter().copied().map(f64::abs).fold(1.0, f64::max);
        assert_slice_close_with_tolerance(
            &plan.inverse(&approx, &detail),
            signal,
            boundary,
            &context,
            reconstruction_tolerance,
        );
    }
}

fn fixture_boundary(mode: &str) -> Boundary {
    match mode {
        "zero" => Boundary::Zero,
        "constant" => Boundary::Constant,
        "symmetric" => Boundary::Symmetric,
        "reflect" => Boundary::Reflect,
        "periodic" => Boundary::Periodic,
        "smooth" => Boundary::Smooth,
        "antisymmetric" => Boundary::Antisymmetric,
        "antireflect" => Boundary::Antireflect,
        "periodization" => Boundary::Periodization,
        unknown => panic!("unknown fixture boundary {unknown}"),
    }
}

fn assert_slice_close(actual: &[f64], expected: &[f64], boundary: Boundary) {
    assert_slice_close_with_tolerance(
        actual,
        expected,
        boundary,
        "static fixture",
        8.0 * f64::EPSILON,
    );
}

fn assert_slice_close_with_tolerance(
    actual: &[f64],
    expected: &[f64],
    boundary: Boundary,
    context: &str,
    tolerance: f64,
) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let element_tolerance = tolerance * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= element_tolerance,
            "{context}: {boundary:?}[{index}]: {actual:.17e} != {expected:.17e} (tol {element_tolerance:.3e})"
        );
    }
}

fn reference_tolerance(wavelet: &Wavelet, boundary: Boundary, signal: &[f64]) -> f64 {
    let signal_scale = signal.iter().copied().map(f64::abs).fold(1.0, f64::max);
    let edge_growth =
        if signal.len() >= 2 && matches!(boundary, Boundary::Smooth | Boundary::Antireflect) {
            let left_slope = (signal[1] - signal[0]).abs();
            let right_slope = (signal[signal.len() - 1] - signal[signal.len() - 2]).abs();
            wavelet.filter_len() as f64 * left_slope.max(right_slope)
        } else {
            0.0
        };
    let extension_scale = signal_scale + edge_growth;
    let filter_norm = wavelet
        .dec_lo()
        .iter()
        .map(|value| value.abs())
        .sum::<f64>()
        .max(wavelet.dec_hi().iter().map(|value| value.abs()).sum());

    // A strict ULP count is ill-defined at coefficients that theoretically
    // vanish: C and Rust may contract arithmetic differently around zero. This
    // is a conservative dot-product forward-error bound instead.
    8.0 * f64::EPSILON * wavelet.filter_len() as f64 * filter_norm * extension_scale
}
