use std::mem::size_of;

use super::super::axis::{
    AxisAnalysisKernel, AxisRowBatch, AxisSynthesisKernel, analyze_axis_tail,
};
use super::super::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::num::forward_axis_fused4_simd;
#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
use crate::num::forward_axis_fused8_simd;
use crate::num::forward_axis_simd;
use crate::simd::AxisAnalysis;
#[test]
fn axis_analysis_kernel_selection_follows_dispatch_backend() {
    let level = SimdLevel::new();
    assert_eq!(
        AxisAnalysisKernel::select(level, 8, size_of::<f32>()),
        AxisAnalysisKernel::Direct
    );
    assert_eq!(
        AxisAnalysisKernel::select(level, 8, size_of::<f64>()),
        AxisAnalysisKernel::Direct
    );

    #[cfg(target_arch = "aarch64")]
    {
        assert_eq!(
            AxisAnalysisKernel::select(level, 48, size_of::<f64>()),
            AxisAnalysisKernel::Fused8
        );
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let dispatch = level.__dispatch_target();
        if dispatch.as_avx512().is_some() {
            assert_eq!(
                AxisAnalysisKernel::select(level, 24, size_of::<f32>()),
                AxisAnalysisKernel::Fused8
            );
            assert_eq!(
                AxisAnalysisKernel::select(level, 16, size_of::<f64>()),
                AxisAnalysisKernel::Fused8
            );
        } else if dispatch.as_avx2().is_some() {
            assert_eq!(
                AxisAnalysisKernel::select(level, 32, size_of::<f32>()),
                AxisAnalysisKernel::Fused4
            );
            assert_eq!(
                AxisAnalysisKernel::select(level, 15, size_of::<f64>()),
                AxisAnalysisKernel::Direct
            );
            assert_eq!(
                AxisAnalysisKernel::select(level, 16, size_of::<f64>()),
                AxisAnalysisKernel::Fused4
            );
        }
    }
}

#[test]
fn axis_row_batch_threshold_follows_dispatch_backend() {
    let level = SimdLevel::new();
    let dispatch = level.__dispatch_target();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if dispatch.as_avx512().is_some() {
        assert!(AxisRowBatch::select(level, 31, size_of::<f64>(), usize::MAX, 1).is_none());
        assert!(AxisRowBatch::select(level, 32, size_of::<f64>(), usize::MAX, 1).is_some());
    } else if dispatch.as_avx2().is_some() {
        assert!(AxisRowBatch::select(level, 19, size_of::<f64>(), usize::MAX, 1).is_none());
        assert!(AxisRowBatch::select(level, 20, size_of::<f64>(), usize::MAX, 1).is_some());
        assert!(AxisRowBatch::select(level, 31, size_of::<f32>(), usize::MAX, 1).is_none());
        assert!(AxisRowBatch::select(level, 32, size_of::<f32>(), usize::MAX, 1).is_some());
    }

    #[cfg(target_arch = "aarch64")]
    if dispatch.as_neon().is_some() {
        assert!(AxisRowBatch::select(level, 47, size_of::<f64>(), usize::MAX, 1).is_none());
        assert!(AxisRowBatch::select(level, 48, size_of::<f64>(), usize::MAX, 1).is_some());
    }
}

#[test]
fn fused_axis_analysis_is_bit_exact_with_direct_analysis() {
    #[cfg(target_arch = "aarch64")]
    {
        assert_fused_axis_analysis_matches_direct::<f32>(AxisAnalysisKernel::Fused8);
        assert_fused_axis_analysis_matches_direct::<f64>(AxisAnalysisKernel::Fused8);
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let level = SimdLevel::new().__dispatch_target();
        if level.as_avx512().is_some() {
            assert_fused_axis_analysis_matches_direct::<f32>(AxisAnalysisKernel::Fused8);
            assert_fused_axis_analysis_matches_direct::<f64>(AxisAnalysisKernel::Fused8);
        } else if level.as_avx2().is_some() {
            assert_fused_axis_analysis_matches_direct::<f32>(AxisAnalysisKernel::Fused4);
            assert_fused_axis_analysis_matches_direct::<f64>(AxisAnalysisKernel::Fused4);
        }
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
fn assert_fused_axis_analysis_matches_direct<T: WaveletNum>(kernel: AxisAnalysisKernel) {
    let wavelet = Wavelet::daubechies(38).unwrap();
    let outer = 2;
    let inner = 19;
    let signal_len = 256;
    let signal: Vec<_> = (0..outer * signal_len * inner)
        .map(|index| T::from_f64(((index * 37 + 11) % 251) as f64 / 37.0 - 3.0))
        .collect();

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
        let plan = create_dwt_plan::<T>(signal_len, &wavelet, boundary, SimdLevel::new()).unwrap();

        let output_len = outer * plan.coeff_len * inner;
        let mut expected_approx = vec![T::zero(); output_len];
        let mut expected_detail = vec![T::zero(); output_len];
        execute_axis_analysis_kernel(
            &plan,
            &signal,
            outer,
            inner,
            &mut expected_approx,
            &mut expected_detail,
            AxisAnalysisKernel::Direct,
        );
        let mut actual_approx = vec![T::zero(); output_len];
        let mut actual_detail = vec![T::zero(); output_len];
        execute_axis_analysis_kernel(
            &plan,
            &signal,
            outer,
            inner,
            &mut actual_approx,
            &mut actual_detail,
            kernel,
        );

        assert_eq!(actual_approx, expected_approx, "{boundary:?} approximation");
        assert_eq!(actual_detail, expected_detail, "{boundary:?} detail");
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
fn execute_axis_analysis_kernel<T: WaveletNum>(
    plan: &PlannedDwt<T>,
    signal: &[T],
    outer: usize,
    inner: usize,
    approx: &mut [T],
    detail: &mut [T],
    kernel: AxisAnalysisKernel,
) {
    let (dec_lo, dec_hi) = plan.filters.analysis();
    let (interior_first_newest, interior_len) =
        plan.analysis.interior.as_ref().map_or((0, 0), |interior| {
            (interior.first_newest, interior.output_len)
        });
    let analysis = AxisAnalysis {
        signal,
        dec_lo,
        dec_hi,
        edge_row_offsets: &plan.analysis.edges.row_offsets,
        edge_terms: &plan.analysis.edges.terms,
        signal_len: plan.signal_len,
        coeff_len: plan.coeff_len,
        outer,
        inner,
        prefix_len: plan.analysis.prefix_len,
        interior_first_newest,
        interior_len,
    };
    let vectorized = match kernel {
        AxisAnalysisKernel::Direct => forward_axis_simd(plan.simd_level, analysis, approx, detail),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        AxisAnalysisKernel::Fused4 => {
            forward_axis_fused4_simd(plan.simd_level, analysis, approx, detail)
        }
        AxisAnalysisKernel::Fused8 => {
            forward_axis_fused8_simd(plan.simd_level, analysis, approx, detail)
        }
    };
    analyze_axis_tail(plan, signal, outer, inner, vectorized, approx, detail);
}

#[test]
fn axis_synthesis_kernel_selection_follows_transform_geometry() {
    assert_eq!(
        AxisSynthesisKernel::select(16, 8, false),
        AxisSynthesisKernel::Direct
    );
    assert_eq!(
        AxisSynthesisKernel::select(2, 76, false),
        AxisSynthesisKernel::Direct
    );
    assert_eq!(
        AxisSynthesisKernel::select(16, 76, true),
        AxisSynthesisKernel::Direct
    );
    assert_eq!(
        AxisSynthesisKernel::select(16, 24, false),
        AxisSynthesisKernel::Batched
    );
}

#[test]
fn axis_execution_matches_independent_signal_execution() {
    let boundaries = [
        Boundary::Zero,
        Boundary::Constant,
        Boundary::Symmetric,
        Boundary::Reflect,
        Boundary::Periodic,
        Boundary::Smooth,
        Boundary::Antisymmetric,
        Boundary::Antireflect,
        Boundary::Periodization,
    ];
    let wavelets = [Wavelet::haar(), Wavelet::daubechies(4).unwrap()];

    for signal_len in [2, 7, 32] {
        for wavelet in &wavelets {
            for &boundary in &boundaries {
                let plan = create_dwt_plan::<f64>(signal_len, wavelet, boundary, SimdLevel::new())
                    .unwrap();
                let outer = 2;
                let inner = 5;
                let signal: Vec<_> = (0..outer * signal_len * inner)
                    .map(|index| {
                        let centered = (index * 37 + 11) % 101;
                        (centered as f64 - 50.0) / 17.0
                    })
                    .collect();
                let mut actual_approx = vec![0.0; outer * plan.coeff_len * inner];
                let mut actual_detail = actual_approx.clone();
                plan.forward_axis_into(
                    &signal,
                    outer,
                    inner,
                    &mut actual_approx,
                    &mut actual_detail,
                    &mut [],
                );

                let mut expected_approx = actual_approx.clone();
                let mut expected_detail = actual_detail.clone();
                for outer_index in 0..outer {
                    for lane in 0..inner {
                        let row: Vec<_> = (0..signal_len)
                            .map(|sample| {
                                signal[(outer_index * signal_len + sample) * inner + lane]
                            })
                            .collect();
                        let (approx, detail) = plan.forward(&row);
                        for coefficient in 0..plan.coeff_len {
                            let output =
                                (outer_index * plan.coeff_len + coefficient) * inner + lane;
                            expected_approx[output] = approx[coefficient];
                            expected_detail[output] = detail[coefficient];
                        }
                    }
                }
                assert_slices_close(&actual_approx, &expected_approx, 2.0e-13);
                assert_slices_close(&actual_detail, &expected_detail, 2.0e-13);

                let mut actual_output = vec![0.0; signal.len()];
                plan.inverse_axis_into(
                    &actual_approx,
                    &actual_detail,
                    outer,
                    inner,
                    &mut actual_output,
                    &mut [],
                );
                let mut expected_output = actual_output.clone();
                for outer_index in 0..outer {
                    for lane in 0..inner {
                        let approx: Vec<_> = (0..plan.coeff_len)
                            .map(|coefficient| {
                                actual_approx
                                    [(outer_index * plan.coeff_len + coefficient) * inner + lane]
                            })
                            .collect();
                        let detail: Vec<_> = (0..plan.coeff_len)
                            .map(|coefficient| {
                                actual_detail
                                    [(outer_index * plan.coeff_len + coefficient) * inner + lane]
                            })
                            .collect();
                        let row = plan.inverse(&approx, &detail);
                        for sample in 0..signal_len {
                            expected_output[(outer_index * signal_len + sample) * inner + lane] =
                                row[sample];
                        }
                    }
                }
                assert_slices_close(&actual_output, &expected_output, 2.0e-13);
            }
        }
    }
}

#[test]
fn batched_axis_inverse_matches_independent_signal_execution() {
    let wavelet = Wavelet::daubechies(38).unwrap();
    for boundary in [Boundary::Symmetric, Boundary::Periodization] {
        let plan = create_dwt_plan::<f64>(16, &wavelet, boundary, SimdLevel::new()).unwrap();
        let outer = 2;
        let inner = 257;
        let coefficient_count = outer * plan.coeff_len * inner;
        let approx: Vec<_> = (0..coefficient_count)
            .map(|index| ((index * 37 + 11) % 251) as f64 / 37.0 - 3.0)
            .collect();
        let detail: Vec<_> = (0..coefficient_count)
            .map(|index| ((index * 41 + 17) % 241) as f64 / 41.0 - 2.5)
            .collect();
        let mut actual = vec![0.0; outer * plan.signal_len * inner];
        plan.inverse_axis_into(&approx, &detail, outer, inner, &mut actual, &mut []);

        let mut expected = actual.clone();
        for outer_index in 0..outer {
            for lane in 0..inner {
                let approx_row: Vec<_> = (0..plan.coeff_len)
                    .map(|coefficient| {
                        approx[(outer_index * plan.coeff_len + coefficient) * inner + lane]
                    })
                    .collect();
                let detail_row: Vec<_> = (0..plan.coeff_len)
                    .map(|coefficient| {
                        detail[(outer_index * plan.coeff_len + coefficient) * inner + lane]
                    })
                    .collect();
                let row = plan.inverse(&approx_row, &detail_row);
                for sample in 0..plan.signal_len {
                    expected[(outer_index * plan.signal_len + sample) * inner + lane] = row[sample];
                }
            }
        }
        assert_slices_close(&actual, &expected, 2.0e-13);
    }
}

#[test]
fn packed_row_analysis_matches_independent_signal_execution() {
    // The packed executor intentionally uses the same tap-order reduction
    // as non-contiguous axis execution. The independent 1-D kernel uses
    // separate even/odd accumulators, so cancellation-heavy f32 boundary
    // rows differ within the normal FIR rounding envelope.
    assert_packed_rows_match::<f32>(|value| value as f64, 3.0e-5, 2.0e-6);
    assert_packed_rows_match::<f64>(|value| value, 2.0e-13, 2.0e-13);
}

fn assert_packed_rows_match<T: WaveletNum>(
    to_f64: fn(T) -> f64,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) {
    let level = SimdLevel::new();
    let wavelet = Wavelet::daubechies(38).unwrap();
    let Some(batch) =
        AxisRowBatch::select(level, wavelet.filter_len(), size_of::<T>(), usize::MAX, 1)
    else {
        return;
    };
    let outer = batch.width + 3;

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
        let plan = create_dwt_plan::<T>(64, &wavelet, boundary, level).unwrap();
        let signal: Vec<_> = (0..outer * plan.signal_len)
            .map(|index| {
                let centered = (index * 37 + 11) % 101;
                T::from_f64((centered as f64 - 50.0) / 17.0)
            })
            .collect();
        let mut actual_approx = vec![T::zero(); outer * plan.coeff_len];
        let mut actual_detail = actual_approx.clone();
        let mut scratch = vec![T::zero(); plan.axis_scratch_len(outer, 1)];
        plan.forward_axis_into(
            &signal,
            outer,
            1,
            &mut actual_approx,
            &mut actual_detail,
            &mut scratch,
        );

        for row in 0..outer {
            let input = &signal[row * plan.signal_len..(row + 1) * plan.signal_len];
            let (expected_approx, expected_detail) = plan.forward(input);
            for coefficient in 0..plan.coeff_len {
                let output = row * plan.coeff_len + coefficient;
                for (actual, expected) in [
                    (actual_approx[output], expected_approx[coefficient]),
                    (actual_detail[output], expected_detail[coefficient]),
                ] {
                    let actual = to_f64(actual);
                    let expected = to_f64(expected);
                    let error = (actual - expected).abs();
                    assert!(
                        error <= absolute_tolerance + relative_tolerance * expected.abs(),
                        "{boundary:?}, row {row}, coefficient {coefficient}: \
                             actual={actual:.17e}, expected={expected:.17e}, error={error:.3e}"
                    );
                }
            }
        }
    }
}

fn assert_slices_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "value {index}: actual={actual:.17e}, expected={expected:.17e}, error={error:.3e}"
        );
    }
}
