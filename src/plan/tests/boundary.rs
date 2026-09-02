use super::super::analysis::for_each_extension_term;
use super::super::*;
use crate::DwtPlanner;

#[test]
fn reflect_rejects_a_single_sample() {
    let mut planner = DwtPlanner::<f64>::new();
    let error = match planner.plan_dwt(1, &Wavelet::haar(), Boundary::Reflect) {
        Ok(_) => panic!("length-one reflect plan unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        WaveletError::BoundaryRequiresLongerSignal { .. }
    ));
}

#[test]
fn antireflect_terms_cover_repeated_reflections() {
    let signal = [1.0_f64, 3.0, 6.0];
    let expected = [
        -29.0, -27.0, -24.0, -21.0, -19.0, -17.0, -14.0, -11.0, -9.0, -7.0, -4.0, -1.0, 1.0, 3.0,
        6.0, 9.0, 11.0, 13.0, 16.0, 19.0, 21.0, 23.0, 26.0, 29.0, 31.0, 33.0, 36.0,
    ];

    let actual: Vec<_> = (-12..=14)
        .map(|index| extended_sample(&signal, index, Boundary::Antireflect))
        .collect();
    assert_eq!(actual, expected);
}

#[test]
fn two_sample_antireflect_is_linear_extrapolation() {
    let signal = [-3.0_f64, 5.0];
    for index in -128..=128 {
        let actual = extended_sample(&signal, index, Boundary::Antireflect);
        let expected = signal[0] + index as f64 * (signal[1] - signal[0]);
        assert_eq!(actual, expected, "extended sample {index}");
    }
}

#[test]
fn compiled_edge_rows_coalesce_repeated_inputs() {
    let wavelet = Wavelet::coiflet(17).unwrap();
    let plan =
        create_dwt_plan::<f64>(16, &wavelet, Boundary::Antireflect, SimdLevel::new()).unwrap();

    for offsets in plan.analysis.edges.row_offsets.windows(2) {
        let row = &plan.analysis.edges.terms[offsets[0]..offsets[1]];
        assert!(row.len() <= plan.signal_len);
        for (position, term) in row.iter().enumerate() {
            assert!(
                row[..position]
                    .iter()
                    .all(|earlier| earlier.input != term.input),
                "edge row contains input {} more than once",
                term.input
            );
        }
    }
}

fn extended_sample(signal: &[f64], index: isize, boundary: Boundary) -> f64 {
    let mut value = 0.0;
    for_each_extension_term::<f64>(index, signal.len(), boundary, |input, weight| {
        value += signal[input] * weight;
    });
    value
}
