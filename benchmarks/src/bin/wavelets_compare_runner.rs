use std::collections::HashSet;
use std::error::Error;
use std::hint::black_box;
use std::io::{self, Read};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use wavelets::{Boundary, DwtPlanner, Level, Wavelet, WaveletNum};

const SCHEMA_VERSION: u32 = 1;
const MAX_BATCH_ITERATIONS: usize = 100_000_000;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: u32,
    config: Config,
    cases: Vec<Case>,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    samples: usize,
    sample_time_ms: f64,
    warmup_batches: usize,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    scope: Scope,
    direction: Direction,
    dtype: Dtype,
    order: usize,
    boundary: String,
    len: usize,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Scope {
    SingleLevel,
    Multilevel,
}

impl Scope {
    const fn as_str(self) -> &'static str {
        match self {
            Self::SingleLevel => "single_level",
            Self::Multilevel => "multilevel",
        }
    }
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Direction {
    Forward,
    Inverse,
}

impl Direction {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Inverse => "inverse",
        }
    }
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Dtype {
    F32,
    F64,
}

impl Dtype {
    const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

#[derive(Serialize)]
struct Response {
    schema: u32,
    engine: Engine,
    results: Vec<BenchmarkResult>,
}

#[derive(Serialize)]
struct Engine {
    name: &'static str,
    language: &'static str,
    clock: &'static str,
    target: String,
    detected_features: Vec<&'static str>,
}

#[derive(Serialize)]
struct BenchmarkResult {
    case_id: String,
    api: &'static str,
    batch_iterations: usize,
    samples_ns: Vec<f64>,
    checksum: f64,
}

trait BenchNum: WaveletNum {
    fn to_f64(self) -> f64;
}

impl BenchNum for f32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl BenchNum for f64 {
    fn to_f64(self) -> f64 {
        self
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let request: Request = serde_json::from_str(&input)?;
    validate_request(&request)?;

    let mut results = Vec::with_capacity(request.cases.len() * 2);
    for case in &request.cases {
        let mut case_results = match case.dtype {
            Dtype::F32 => run_case::<f32>(case, request.config)?,
            Dtype::F64 => run_case::<f64>(case, request.config)?,
        };
        results.append(&mut case_results);
    }

    let response = Response {
        schema: SCHEMA_VERSION,
        engine: Engine {
            name: "wavelets",
            language: "Rust",
            clock: "std::time::Instant",
            target: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
            detected_features: detected_features(),
        },
        results,
    };
    serde_json::to_writer(io::stdout().lock(), &response)?;
    Ok(())
}

fn validate_request(request: &Request) -> Result<(), String> {
    if request.schema != SCHEMA_VERSION {
        return Err(format!(
            "unsupported request schema {}; expected {SCHEMA_VERSION}",
            request.schema
        ));
    }
    if request.config.samples < 3 {
        return Err("at least three samples are required".into());
    }
    if !request.config.sample_time_ms.is_finite() || request.config.sample_time_ms <= 0.0 {
        return Err("sample_time_ms must be finite and positive".into());
    }
    if request.cases.is_empty() {
        return Err("at least one benchmark case is required".into());
    }

    let mut ids = HashSet::with_capacity(request.cases.len());
    for case in &request.cases {
        if case.len == 0 {
            return Err(format!("{} has an empty signal", case.id));
        }
        if !(1..=38).contains(&case.order) {
            return Err(format!("{} has unsupported db{}", case.id, case.order));
        }
        boundary(&case.boundary)?;
        let expected_id = format!(
            "{}/{}/{}/db{}/{}/{}",
            case.scope.as_str(),
            case.direction.as_str(),
            case.dtype.as_str(),
            case.order,
            case.boundary,
            case.len
        );
        if case.id != expected_id {
            return Err(format!(
                "case id {:?} does not match {:?}",
                case.id, expected_id
            ));
        }
        if !ids.insert(&case.id) {
            return Err(format!("duplicate case id {:?}", case.id));
        }
    }
    Ok(())
}

fn run_case<T: BenchNum>(
    case: &Case,
    config: Config,
) -> Result<Vec<BenchmarkResult>, Box<dyn Error>> {
    match case.scope {
        Scope::SingleLevel => run_single_level::<T>(case, config),
        Scope::Multilevel => run_multilevel::<T>(case, config),
    }
}

fn run_single_level<T: BenchNum>(
    case: &Case,
    config: Config,
) -> Result<Vec<BenchmarkResult>, Box<dyn Error>> {
    let signal = signal::<T>(case.len);
    let wavelet = Wavelet::daubechies(case.order)?;
    let mut planner = DwtPlanner::<T>::new();
    let plan = planner.plan_dwt(case.len, &wavelet, boundary(&case.boundary)?)?;
    let mut approx = vec![T::zero(); plan.coeff_len()];
    let mut detail = vec![T::zero(); plan.coeff_len()];
    let mut output = vec![T::zero(); plan.signal_len()];
    let mut scratch = vec![T::zero(); plan.scratch_len()];
    plan.forward_into(&signal, &mut approx, &mut detail, &mut scratch);

    let (into_batch_iterations, into_samples) = match case.direction {
        Direction::Forward => measure(
            || {
                plan.forward_into(black_box(&signal), &mut approx, &mut detail, &mut scratch);
                black_box((&approx, &detail));
            },
            config,
        ),
        Direction::Inverse => measure(
            || {
                plan.inverse_into(
                    black_box(&approx),
                    black_box(&detail),
                    &mut output,
                    &mut scratch,
                );
                black_box(&output);
            },
            config,
        ),
    };
    let into_checksum = match case.direction {
        Direction::Forward => checksum(&approx) + checksum(&detail),
        Direction::Inverse => checksum(&output),
    };

    let allocating_checksum = match case.direction {
        Direction::Forward => {
            let (allocated_approx, allocated_detail) = plan.forward(&signal);
            checksum(&allocated_approx) + checksum(&allocated_detail)
        }
        Direction::Inverse => checksum(&plan.inverse(&approx, &detail)),
    };
    let (allocating_batch_iterations, allocating_samples) = match case.direction {
        Direction::Forward => measure(
            || {
                black_box(plan.forward(black_box(&signal)));
            },
            config,
        ),
        Direction::Inverse => measure(
            || {
                black_box(plan.inverse(black_box(&approx), black_box(&detail)));
            },
            config,
        ),
    };

    Ok(vec![
        BenchmarkResult {
            case_id: case.id.clone(),
            api: "into",
            batch_iterations: into_batch_iterations,
            samples_ns: into_samples,
            checksum: into_checksum,
        },
        BenchmarkResult {
            case_id: case.id.clone(),
            api: "allocating",
            batch_iterations: allocating_batch_iterations,
            samples_ns: allocating_samples,
            checksum: allocating_checksum,
        },
    ])
}

fn run_multilevel<T: BenchNum>(
    case: &Case,
    config: Config,
) -> Result<Vec<BenchmarkResult>, Box<dyn Error>> {
    let signal = signal::<T>(case.len);
    let wavelet = Wavelet::daubechies(case.order)?;
    let mut planner = DwtPlanner::<T>::new();
    let plan = planner.plan_wavedec(case.len, &wavelet, boundary(&case.boundary)?, Level::Max)?;
    let mut decomposition = plan.allocate_decomposition();
    let mut output = vec![T::zero(); plan.signal_len()];
    let mut scratch = vec![T::zero(); plan.scratch_len()];
    plan.forward_into(&signal, &mut decomposition, &mut scratch);

    let (into_batch_iterations, into_samples) = match case.direction {
        Direction::Forward => measure(
            || {
                plan.forward_into(black_box(&signal), &mut decomposition, &mut scratch);
                black_box(decomposition.as_slice());
            },
            config,
        ),
        Direction::Inverse => measure(
            || {
                plan.inverse_into(black_box(&decomposition), &mut output, &mut scratch);
                black_box(&output);
            },
            config,
        ),
    };
    let into_checksum = match case.direction {
        Direction::Forward => checksum(decomposition.as_slice()),
        Direction::Inverse => checksum(&output),
    };

    let allocating_checksum = match case.direction {
        Direction::Forward => checksum(plan.forward(&signal).as_slice()),
        Direction::Inverse => checksum(&plan.inverse(&decomposition)),
    };
    let (allocating_batch_iterations, allocating_samples) = match case.direction {
        Direction::Forward => measure(
            || {
                black_box(plan.forward(black_box(&signal)));
            },
            config,
        ),
        Direction::Inverse => measure(
            || {
                black_box(plan.inverse(black_box(&decomposition)));
            },
            config,
        ),
    };

    Ok(vec![
        BenchmarkResult {
            case_id: case.id.clone(),
            api: "into",
            batch_iterations: into_batch_iterations,
            samples_ns: into_samples,
            checksum: into_checksum,
        },
        BenchmarkResult {
            case_id: case.id.clone(),
            api: "allocating",
            batch_iterations: allocating_batch_iterations,
            samples_ns: allocating_samples,
            checksum: allocating_checksum,
        },
    ])
}

fn measure(mut operation: impl FnMut(), config: Config) -> (usize, Vec<f64>) {
    let target = Duration::from_secs_f64(config.sample_time_ms / 1_000.0);
    let batch_iterations = calibrate(&mut operation, target);
    for _ in 0..config.warmup_batches {
        run_batch(&mut operation, batch_iterations);
    }

    let samples = (0..config.samples)
        .map(|_| {
            let elapsed = run_batch(&mut operation, batch_iterations);
            elapsed.as_secs_f64() * 1.0e9 / batch_iterations as f64
        })
        .collect();
    (batch_iterations, samples)
}

fn calibrate(operation: &mut impl FnMut(), target: Duration) -> usize {
    let minimum = target / 4;
    let mut iterations = 1_usize;
    loop {
        let elapsed = run_batch(operation, iterations);
        if elapsed >= minimum || iterations >= MAX_BATCH_ITERATIONS {
            let elapsed_seconds = elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
            let estimated =
                (iterations as f64 * target.as_secs_f64() / elapsed_seconds).ceil() as usize;
            return estimated.clamp(1, MAX_BATCH_ITERATIONS);
        }
        iterations = iterations.saturating_mul(2).min(MAX_BATCH_ITERATIONS);
    }
}

fn run_batch(operation: &mut impl FnMut(), iterations: usize) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        operation();
    }
    start.elapsed()
}

fn signal<T: WaveletNum>(len: usize) -> Vec<T> {
    (0..len)
        .map(|index| {
            let primary = ((index * 17) % 257) as i32 - 128;
            let secondary = (index % 11) as i32 - 5;
            T::from_f64(f64::from(primary) / 64.0 + f64::from(secondary) / 16.0)
        })
        .collect()
}

fn checksum<T: BenchNum>(values: &[T]) -> f64 {
    values.iter().map(|value| value.to_f64().abs()).sum()
}

fn boundary(name: &str) -> Result<Boundary, String> {
    match name {
        "zero" => Ok(Boundary::Zero),
        "constant" => Ok(Boundary::Constant),
        "symmetric" => Ok(Boundary::Symmetric),
        "reflect" => Ok(Boundary::Reflect),
        "periodic" => Ok(Boundary::Periodic),
        "smooth" => Ok(Boundary::Smooth),
        "antisymmetric" => Ok(Boundary::Antisymmetric),
        "antireflect" => Ok(Boundary::Antireflect),
        "periodization" => Ok(Boundary::Periodization),
        unknown => Err(format!("unsupported boundary {unknown:?}")),
    }
}

fn detected_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            features.push("avx2");
        }
        if std::arch::is_x86_feature_detected!("fma") {
            features.push("fma");
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        features.push("neon");
    }
    features
}
