use std::collections::HashMap;
#[cfg(feature = "experimental-kernels")]
use std::sync::Arc;

#[cfg(feature = "experimental-kernels")]
use fearless_simd::Level as SimdLevel;

#[cfg(feature = "experimental-kernels")]
use crate::lattice::LatticeFilter;
#[cfg(feature = "experimental-kernels")]
use crate::simd::MIN_LATTICE_OUTPUTS;
use crate::{Boundary, Wavelet, WaveletNum};

#[cfg(feature = "experimental-kernels")]
use super::annihilator::{AnnihilatorAnalysis, AnnihilatorFilter};
use super::{Butterfly, F64Butterfly};

// Materializing two planes does not repay its setup cost below six taps. SIMD
// interiors are also substantially cheaper than scalar boundary rows, so the
// planar executor is retained only while edges are a meaningful share of the
// work.
const MIN_PLANAR_FILTER_LEN: usize = 6;
const MAX_PLANAR_INTERIORS_PER_EDGE: usize = 6;

#[derive(Debug)]
pub(super) struct InteriorAnalysis<T> {
    pub(super) first_newest: usize,
    pub(super) output_len: usize,
    pub(super) kernel: AnalysisKernel<T>,
}

#[derive(Clone, Debug)]
pub(super) enum AnalysisKernel<T> {
    Direct,
    Butterfly {
        low_scale: T,
        high_scale: T,
    },
    #[cfg(feature = "experimental-kernels")]
    Lattice(Arc<LatticeFilter<T>>),
}

#[derive(Debug)]
pub(super) struct AnalysisPlan<T> {
    pub(super) edges: EdgePlan<T>,
    pub(super) prefix_len: usize,
    pub(super) interior: Option<InteriorAnalysis<T>>,
    pub(super) materialized: Option<MaterializedAnalysis<T>>,
    #[cfg(feature = "experimental-kernels")]
    pub(super) annihilator: Option<AnnihilatorAnalysis<T>>,
}

pub(super) struct AnalysisBackends<T> {
    pub(super) butterfly: Option<Butterfly<T>>,
    #[cfg(feature = "experimental-kernels")]
    pub(super) annihilator: Option<Arc<AnnihilatorFilter<T>>>,
    #[cfg(feature = "experimental-kernels")]
    pub(super) lattice: Option<Arc<LatticeFilter<T>>>,
}

#[derive(Debug)]
pub(super) struct EdgePlan<T> {
    // Each row is the filter-composed finite-boundary transform for one
    // approximation/detail coefficient pair. Both channels share every input
    // load, and repeated references to the same finite sample are coalesced.
    pub(super) row_offsets: Box<[usize]>,
    pub(super) terms: Box<[EdgeTerm<T>]>,
    ordered_rules: Box<[SampleRule<T>]>,
}

#[derive(Debug)]
pub(super) struct MaterializedAnalysis<T> {
    rules: Box<[SampleRule<T>]>,
    pub(super) even_len: usize,
    pub(super) first_newest: usize,
}

impl<T> MaterializedAnalysis<T> {
    pub(super) fn len(&self) -> usize {
        self.rules.len()
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct EdgeTerm<T> {
    pub(crate) input: usize,
    pub(crate) low: T,
    pub(crate) high: T,
}

#[derive(Clone, Copy, Debug)]
enum SampleRule<T> {
    Zero,
    Direct {
        index: usize,
        negative: bool,
    },
    Smooth {
        edge: usize,
        neighbor: usize,
        distance: usize,
    },
    Linear2 {
        indices: [usize; 2],
        weights: [T; 2],
    },
    Linear3 {
        indices: [usize; 3],
        weights: [T; 3],
    },
}

pub(super) fn analysis_butterfly(wavelet: &Wavelet) -> Option<F64Butterfly> {
    // Detect the matrix factorization itself rather than a built-in name so
    // equivalent caller-supplied banks select the same kernel.
    let [low_first, low_second] = wavelet.dec_lo() else {
        return None;
    };
    let [high_first, high_second] = wavelet.dec_hi() else {
        return None;
    };
    (*low_first == *low_second && *high_first == -*high_second).then_some(F64Butterfly {
        low_scale: *low_first,
        high_scale: *high_second,
    })
}

#[cfg(feature = "experimental-kernels")]
pub(super) fn lattice_simd_supported(level: SimdLevel) -> bool {
    let level = level.__dispatch_target();
    #[cfg(target_arch = "aarch64")]
    {
        !level.is_fallback()
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        level.as_avx512().is_some()
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    {
        let _ = level;
        false
    }
}

#[cfg(feature = "experimental-kernels")]
pub(super) fn lattice_preempts_annihilator(level: SimdLevel) -> bool {
    let level = level.__dispatch_target();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // On AVX-512, the lattice is faster even than the annihilator's
        // zero-event endpoint. Scanning for structure can therefore never
        // select a better executor once this backend is available.
        level.as_avx512().is_some()
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let _ = level;
        false
    }
}

pub(crate) fn coefficient_len(signal_len: usize, filter_len: usize, boundary: Boundary) -> usize {
    if boundary == Boundary::Periodization {
        signal_len.div_ceil(2)
    } else {
        (signal_len + filter_len - 1) / 2
    }
}

fn select_analysis_kernel<T: WaveletNum>(
    boundary: Boundary,
    output_len: usize,
    backends: &AnalysisBackends<T>,
) -> AnalysisKernel<T> {
    if let Some(butterfly) = backends.butterfly {
        return AnalysisKernel::Butterfly {
            low_scale: butterfly.low_scale,
            high_scale: butterfly.high_scale,
        };
    }
    #[cfg(feature = "experimental-kernels")]
    if boundary != Boundary::Periodization
        && output_len >= MIN_LATTICE_OUTPUTS
        && let Some(lattice) = &backends.lattice
    {
        return AnalysisKernel::Lattice(lattice.clone());
    }
    #[cfg(not(feature = "experimental-kernels"))]
    let _ = (boundary, output_len);
    AnalysisKernel::Direct
}

pub(super) fn build_analysis<T: WaveletNum>(
    signal_len: usize,
    coeff_len: usize,
    dec_lo: &[T],
    dec_hi: &[T],
    backends: AnalysisBackends<T>,
    boundary: Boundary,
    simd_available: bool,
) -> AnalysisPlan<T> {
    debug_assert_eq!(dec_lo.len(), dec_hi.len());
    let filter_len = dec_lo.len();
    let phase = if boundary == Boundary::Periodization {
        filter_len / 2
    } else {
        1
    };
    let is_interior = |coefficient: usize| {
        let newest = (2 * coefficient + phase) as isize;
        let oldest = newest - (filter_len - 1) as isize;
        oldest >= 0 && newest < signal_len as isize
    };
    let interior_start = (0..coeff_len).find(|&coefficient| is_interior(coefficient));
    let interior_end = interior_start.map_or(coeff_len, |start| {
        start
            + (start..coeff_len)
                .take_while(|&coefficient| is_interior(coefficient))
                .count()
    });
    debug_assert!((interior_end..coeff_len).all(|coefficient| !is_interior(coefficient)));

    let prefix_end = interior_start.unwrap_or(coeff_len);
    let edge_count = prefix_end + coeff_len - interior_end;
    let mut row_offsets = Vec::with_capacity(edge_count + 1);
    let mut terms = Vec::<EdgeTerm<T>>::with_capacity(edge_count * filter_len);
    let mut ordered_rules = Vec::with_capacity(edge_count * filter_len);
    // A dense position map avoids hashing when it is no larger than the raw
    // edge-rule grid. Large signals retain an O(filter_len) sparse planner.
    let dense_position_limit = edge_count.saturating_mul(filter_len);
    let mut dense_positions =
        (signal_len <= dense_position_limit).then(|| vec![usize::MAX; signal_len]);
    let mut sparse_positions = dense_positions
        .is_none()
        .then(|| HashMap::<usize, usize>::with_capacity(3 * filter_len));
    row_offsets.push(0);
    for coefficient in (0..prefix_end).chain(interior_end..coeff_len) {
        if let Some(positions) = &mut sparse_positions {
            positions.clear();
        }
        let row_start = terms.len();
        let newest = (2 * coefficient + phase) as isize;
        for tap in 0..filter_len {
            let extended_index = newest - tap as isize;
            ordered_rules.push(extension_rule::<T>(extended_index, signal_len, boundary));
            for_each_extension_term(extended_index, signal_len, boundary, |input, weight| {
                let low = dec_lo[tap] * weight;
                let high = dec_hi[tap] * weight;
                let position = if let Some(positions) = &mut dense_positions {
                    let position = positions[input];
                    if position != usize::MAX && position >= row_start {
                        Some(position)
                    } else {
                        positions[input] = terms.len();
                        None
                    }
                } else {
                    let positions = sparse_positions
                        .as_mut()
                        .expect("one edge position map is always available");
                    if let Some(&position) = positions.get(&input) {
                        Some(position)
                    } else {
                        positions.insert(input, terms.len());
                        None
                    }
                };
                if let Some(position) = position {
                    terms[position].low += low;
                    terms[position].high += high;
                } else {
                    terms.push(EdgeTerm { input, low, high });
                }
            });
        }
        row_offsets.push(terms.len());
    }
    let interior = interior_start.map(|start| InteriorAnalysis {
        first_newest: 2 * start + phase,
        output_len: interior_end - start,
        kernel: select_analysis_kernel(boundary, interior_end - start, &backends),
    });
    let direct_interior = interior
        .as_ref()
        .is_none_or(|interior| matches!(&interior.kernel, AnalysisKernel::Direct));
    let interior_len = interior_end - prefix_end;
    let materialized = (simd_available
        && direct_interior
        && filter_len >= MIN_PLANAR_FILTER_LEN
        && edge_count.saturating_mul(MAX_PLANAR_INTERIORS_PER_EDGE) >= interior_len)
        .then(|| {
            let first_extended_index = phase as isize - (filter_len - 1) as isize;
            let extension_len = coeff_len
                .saturating_sub(1)
                .checked_mul(2)
                .and_then(|len| len.checked_add(filter_len))
                .expect("analysis extension length overflow");
            let even_len = extension_len.div_ceil(2);
            MaterializedAnalysis {
                // A filter tap visits one parity of the extension at consecutive
                // locations. Storing both parities separately turns those visits
                // into contiguous SIMD loads without changing tap order.
                rules: (0..extension_len)
                    .step_by(2)
                    .chain((1..extension_len).step_by(2))
                    .map(|offset| {
                        extension_rule::<T>(
                            first_extended_index + offset as isize,
                            signal_len,
                            boundary,
                        )
                    })
                    .collect(),
                even_len,
                first_newest: filter_len - 1,
            }
        });

    AnalysisPlan {
        edges: EdgePlan {
            row_offsets: row_offsets.into_boxed_slice(),
            terms: terms.into_boxed_slice(),
            ordered_rules: ordered_rules.into_boxed_slice(),
        },
        prefix_len: prefix_end,
        interior,
        materialized,
        #[cfg(feature = "experimental-kernels")]
        annihilator: backends
            .annihilator
            .map(|filter| AnnihilatorAnalysis::new(signal_len, coeff_len, boundary, filter)),
    }
}

pub(super) fn materialize_extension<T: WaveletNum>(
    signal: &[T],
    analysis: &MaterializedAnalysis<T>,
    out: &mut [T],
) {
    debug_assert_eq!(out.len(), analysis.rules.len());
    for (sample, rule) in out.iter_mut().zip(analysis.rules.iter().copied()) {
        *sample = evaluate_sample(rule, &mut |input| signal[input]);
    }
}

#[inline]
pub(super) fn analyze_planar<T: WaveletNum>(
    even: &[T],
    odd: &[T],
    newest: usize,
    dec_lo: &[T],
    dec_hi: &[T],
) -> (T, T) {
    let mut low = T::zero();
    let mut high = T::zero();
    for tap in 0..dec_lo.len() {
        let position = newest - tap;
        let plane = if position.is_multiple_of(2) {
            even
        } else {
            odd
        };
        let sample = plane[position / 2];
        low += dec_lo[tap] * sample;
        high += dec_hi[tap] * sample;
    }
    (low, high)
}

#[inline]
pub(super) fn analyze_interior<T: WaveletNum>(
    signal: &[T],
    newest: usize,
    dec_lo: &[T],
    dec_hi: &[T],
) -> (T, T) {
    let mut low = T::zero();
    let mut high = T::zero();
    for tap in 0..dec_lo.len() {
        let sample = signal[newest - tap];
        low += dec_lo[tap] * sample;
        high += dec_hi[tap] * sample;
    }
    (low, high)
}

#[inline]
pub(super) fn analyze_edges<T: WaveletNum>(
    signal: &[T],
    edges: &EdgePlan<T>,
    first_row: usize,
    dec_lo: &[T],
    dec_hi: &[T],
    approx: &mut [T],
    detail: &mut [T],
) {
    debug_assert_eq!(approx.len(), detail.len());
    debug_assert!(first_row + approx.len() < edges.row_offsets.len());
    for (row, (approximation, detail)) in
        (first_row..).zip(approx.iter_mut().zip(detail.iter_mut()))
    {
        (*approximation, *detail) =
            analyze_edge_row(edges, row, dec_lo, dec_hi, |input| signal[input]);
    }
}

#[inline]
pub(super) fn analyze_edge_row<T: WaveletNum>(
    edges: &EdgePlan<T>,
    row: usize,
    dec_lo: &[T],
    dec_hi: &[T],
    sample: impl FnMut(usize) -> T,
) -> (T, T) {
    let start = row * dec_lo.len();
    analyze_edge_rules(
        &edges.ordered_rules[start..start + dec_lo.len()],
        dec_lo,
        dec_hi,
        sample,
    )
}

#[inline]
fn analyze_edge_rules<T: WaveletNum>(
    rules: &[SampleRule<T>],
    dec_lo: &[T],
    dec_hi: &[T],
    mut sample: impl FnMut(usize) -> T,
) -> (T, T) {
    debug_assert_eq!(rules.len(), dec_lo.len());
    debug_assert_eq!(rules.len(), dec_hi.len());
    let mut low = T::zero();
    let mut high = T::zero();
    for (tap, rule) in rules.iter().copied().enumerate() {
        let sample = evaluate_sample(rule, &mut sample);
        low += dec_lo[tap] * sample;
        high += dec_hi[tap] * sample;
    }
    (low, high)
}

fn extension_rule<T: WaveletNum>(
    index: isize,
    signal_len: usize,
    boundary: Boundary,
) -> SampleRule<T> {
    if (0..signal_len as isize).contains(&index) {
        return SampleRule::Direct {
            index: index as usize,
            negative: false,
        };
    }

    if boundary == Boundary::Smooth {
        if signal_len == 1 {
            return SampleRule::Direct {
                index: 0,
                negative: false,
            };
        }
        return SampleRule::Smooth {
            edge: if index < 0 { 0 } else { signal_len - 1 },
            neighbor: if index < 0 { 1 } else { signal_len - 2 },
            distance: if index < 0 {
                (-index) as usize
            } else {
                (index - (signal_len - 1) as isize) as usize
            },
        };
    }

    let mut indices = [0; 3];
    let mut weights = [T::zero(); 3];
    let mut len = 0;
    for_each_extension_term(index, signal_len, boundary, |input, weight| {
        indices[len] = input;
        weights[len] = weight;
        len += 1;
    });
    match len {
        0 => SampleRule::Zero,
        1 if weights[0] == T::from_f64(1.0) => SampleRule::Direct {
            index: indices[0],
            negative: false,
        },
        1 if weights[0] == T::from_f64(-1.0) => SampleRule::Direct {
            index: indices[0],
            negative: true,
        },
        2 => SampleRule::Linear2 {
            indices: [indices[0], indices[1]],
            weights: [weights[0], weights[1]],
        },
        3 => SampleRule::Linear3 { indices, weights },
        _ => unreachable!("one extension sample has at most three finite terms"),
    }
}

#[inline]
fn evaluate_sample<T: WaveletNum>(rule: SampleRule<T>, sample: &mut impl FnMut(usize) -> T) -> T {
    match rule {
        SampleRule::Zero => T::zero(),
        SampleRule::Direct { index, negative } => {
            let value = sample(index);
            if negative { T::zero() - value } else { value }
        }
        SampleRule::Smooth {
            edge,
            neighbor,
            distance,
        } => {
            let edge = sample(edge);
            edge + T::from_f64(distance as f64) * (edge - sample(neighbor))
        }
        SampleRule::Linear2 { indices, weights } => {
            sample(indices[0]) * weights[0] + sample(indices[1]) * weights[1]
        }
        SampleRule::Linear3 { indices, weights } => {
            sample(indices[0]) * weights[0]
                + sample(indices[1]) * weights[1]
                + sample(indices[2]) * weights[2]
        }
    }
}

pub(super) fn for_each_extension_term<T: WaveletNum>(
    index: isize,
    signal_len: usize,
    boundary: Boundary,
    mut visit: impl FnMut(usize, T),
) {
    // Every supported extension is a linear map from the finite signal to one
    // requested sample. Planning composes these terms with both analysis
    // filters; execution never needs to know which boundary mode produced it.
    let weight = |value: f64| T::from_f64(value);
    if (0..signal_len as isize).contains(&index) {
        visit(index as usize, weight(1.0));
        return;
    }

    match boundary {
        Boundary::Zero => {}
        Boundary::Constant => {
            visit(if index < 0 { 0 } else { signal_len - 1 }, weight(1.0));
        }
        Boundary::Periodic => {
            visit(index.rem_euclid(signal_len as isize) as usize, weight(1.0));
        }
        Boundary::Periodization => {
            let periodic_len = signal_len + signal_len % 2;
            let wrapped = index.rem_euclid(periodic_len as isize) as usize;
            visit(wrapped.min(signal_len - 1), weight(1.0));
        }
        Boundary::Symmetric => {
            let period = 2 * signal_len;
            let wrapped = index.rem_euclid(period as isize) as usize;
            let reflected = if wrapped < signal_len {
                wrapped
            } else {
                period - 1 - wrapped
            };
            visit(reflected, weight(1.0));
        }
        Boundary::Antisymmetric => {
            let period = 2 * signal_len;
            let wrapped = index.rem_euclid(period as isize) as usize;
            if wrapped < signal_len {
                visit(wrapped, weight(1.0));
            } else {
                visit(period - 1 - wrapped, weight(-1.0));
            }
        }
        Boundary::Reflect => {
            let span = signal_len - 1;
            let period = 2 * span;
            let wrapped = index.rem_euclid(period as isize) as usize;
            let reflected = if wrapped < signal_len {
                wrapped
            } else {
                period - wrapped
            };
            visit(reflected, weight(1.0));
        }
        Boundary::Smooth => {
            if signal_len == 1 {
                visit(0, weight(1.0));
            } else if index < 0 {
                let distance = (-index) as f64;
                visit(0, weight(1.0 + distance));
                visit(1, weight(-distance));
            } else {
                let distance = (index - (signal_len - 1) as isize) as f64;
                visit(signal_len - 1, weight(1.0 + distance));
                visit(signal_len - 2, weight(-distance));
            }
        }
        Boundary::Antireflect => for_each_antireflect_term(index, signal_len, visit),
    }
}

fn for_each_antireflect_term<T: WaveletNum>(
    index: isize,
    signal_len: usize,
    mut visit: impl FnMut(usize, T),
) {
    debug_assert!(signal_len >= 2);
    debug_assert!(!(0..signal_len as isize).contains(&index));

    let last = signal_len - 1;
    let distance = if index < 0 {
        (-index) as usize
    } else {
        index as usize - last
    };
    let segment = (distance - 1) / last;
    let offset = (distance - 1) % last + 1;
    let weight = |value: isize| T::from_f64(value as f64);

    if index < 0 {
        if segment == 0 {
            visit(0, weight(2));
            visit(offset, weight(-1));
        } else if segment.is_multiple_of(2) {
            visit(0, weight(segment as isize + 2));
            visit(last, weight(-(segment as isize)));
            visit(offset, weight(-1));
        } else {
            visit(0, weight(segment as isize + 1));
            visit(last, weight(-(segment as isize) - 1));
            visit(last - offset, weight(1));
        }
    } else if segment == 0 {
        visit(last, weight(2));
        visit(last - offset, weight(-1));
    } else if segment.is_multiple_of(2) {
        visit(0, weight(-(segment as isize)));
        visit(last, weight(segment as isize + 2));
        visit(last - offset, weight(-1));
    } else {
        visit(0, weight(-(segment as isize) - 1));
        visit(last, weight(segment as isize + 1));
        visit(offset, weight(1));
    }
}
