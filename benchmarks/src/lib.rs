use wavelets::{Boundary, Wavelet, WaveletNum};

pub const SIGNAL_LENGTHS: [usize; 5] = [64, 256, 1_024, 4_096, 16_384];
pub const FILTER_WAVELETS: [&str; 8] = [
    "db1", "db4", "db20", "db38", "sym4", "coif3", "bior4.4", "rbio4.4",
];
pub const BOUNDARIES: [(&str, Boundary); 9] = [
    ("zero", Boundary::Zero),
    ("constant", Boundary::Constant),
    ("symmetric", Boundary::Symmetric),
    ("reflect", Boundary::Reflect),
    ("periodic", Boundary::Periodic),
    ("smooth", Boundary::Smooth),
    ("antisymmetric", Boundary::Antisymmetric),
    ("antireflect", Boundary::Antireflect),
    ("periodization", Boundary::Periodization),
];

#[derive(Clone, Copy)]
pub struct Case {
    pub len: usize,
    pub wavelet_name: &'static str,
    pub boundary_name: &'static str,
    pub boundary: Boundary,
}

pub fn representative_cases() -> Vec<Case> {
    let mut cases = Vec::new();

    for len in SIGNAL_LENGTHS {
        push_unique(
            &mut cases,
            Case {
                len,
                wavelet_name: "db4",
                boundary_name: "symmetric",
                boundary: Boundary::Symmetric,
            },
        );
    }
    for wavelet_name in FILTER_WAVELETS {
        push_unique(
            &mut cases,
            Case {
                len: 4_096,
                wavelet_name,
                boundary_name: "symmetric",
                boundary: Boundary::Symmetric,
            },
        );
    }
    for (boundary_name, boundary) in BOUNDARIES {
        push_unique(
            &mut cases,
            Case {
                len: 4_096,
                wavelet_name: "db4",
                boundary_name,
                boundary,
            },
        );
    }

    cases
}

pub fn wavelet(name: &str) -> Wavelet {
    Wavelet::from_name(name).expect("benchmark wavelet is supported")
}

pub fn signal<T: WaveletNum>(len: usize) -> Vec<T> {
    (0..len)
        .map(|index| {
            let x = index as f64;
            T::from_f64((x * 0.013).sin() + 0.25 * (x * 0.071).cos() + (index % 17) as f64 / 17.0)
        })
        .collect()
}

fn push_unique(cases: &mut Vec<Case>, candidate: Case) {
    if !cases.iter().any(|case| {
        case.len == candidate.len
            && case.wavelet_name == candidate.wavelet_name
            && case.boundary == candidate.boundary
    }) {
        cases.push(candidate);
    }
}
