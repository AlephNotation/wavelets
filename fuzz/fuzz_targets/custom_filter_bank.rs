#![no_main]

use libfuzzer_sys::fuzz_target;
use wavelets::{Boundary, DwtPlanner, Wavelet, WaveletError};
use wavelets_fuzz::{assert_same_f64, boundary, decode_filter_coefficients, make_even_nonempty};

fuzz_target!(|data: &[u8]| {
    let shape = data.first().copied().unwrap_or(0) % 6;
    let selected_boundary = boundary(data.get(1).copied().unwrap_or(0));
    let signal_len = usize::from(data.get(2).copied().unwrap_or(0) % 65);
    let mut dec_lo = decode_filter_coefficients(data.get(3..).unwrap_or_default());
    make_even_nonempty(&mut dec_lo);
    let mut dec_hi: Vec<_> = dec_lo.iter().rev().copied().collect();
    let rec_lo: Vec<_> = dec_lo.iter().rev().copied().collect();
    let rec_hi: Vec<_> = dec_hi.iter().rev().copied().collect();

    match shape {
        0 | 5 => {}
        1 => {
            dec_lo.clear();
            dec_hi.clear();
        }
        2 => {
            dec_lo.push(0.0);
            dec_hi.push(0.0);
        }
        3 => {
            dec_hi.pop();
        }
        4 => {
            dec_lo[0] = f64::NAN;
        }
        _ => unreachable!("shape is a remainder modulo six"),
    }

    let wavelet = Wavelet::from_filters(&dec_lo, &dec_hi, &rec_lo, &rec_hi);
    if !matches!(shape, 0 | 5) {
        assert!(wavelet.is_err());
        return;
    }

    let wavelet = wavelet.expect("normalized valid filter bank is accepted");
    let signal: Vec<_> = (0..signal_len)
        .map(|index| f64::from((index as i16).wrapping_mul(257)) / 32768.0)
        .collect();
    let mut planner = DwtPlanner::<f64>::new();
    let plan = planner.plan_dwt(signal_len, &wavelet, selected_boundary);
    if signal_len == 0 {
        assert!(matches!(plan, Err(WaveletError::EmptySignal)));
        return;
    }
    if signal_len == 1 && matches!(selected_boundary, Boundary::Reflect | Boundary::Antireflect) {
        assert!(matches!(
            plan,
            Err(WaveletError::BoundaryRequiresLongerSignal { .. })
        ));
        return;
    }

    let plan = plan.expect("normalized custom filter plan is valid");
    let (approx, detail) = plan.forward(&signal);
    let mut approx_into = vec![0.0; plan.coeff_len()];
    let mut detail_into = vec![0.0; plan.coeff_len()];
    let mut scratch = vec![0.0; plan.scratch_len()];
    plan.forward_into(&signal, &mut approx_into, &mut detail_into, &mut scratch);
    assert_same_f64(&approx_into, &approx);
    assert_same_f64(&detail_into, &detail);

    let reconstructed = plan.inverse(&approx, &detail);
    let mut reconstructed_into = vec![0.0; signal_len];
    plan.inverse_into(&approx, &detail, &mut reconstructed_into, &mut scratch);
    assert_same_f64(&reconstructed_into, &reconstructed);
});
