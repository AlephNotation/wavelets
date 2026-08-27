#![deny(unsafe_code)]

use std::fmt::Display;
use std::sync::Arc;

use numpy::{Element, IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use wavelets::{Boundary, Decomposition, Dwt, DwtPlanner, Level, WavedecPlan, Wavelet, WaveletNum};

fn value_error(error: impl Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn contiguous_slice<'array, T: Element>(
    array: &'array PyReadonlyArray1<'_, T>,
) -> PyResult<&'array [T]> {
    array.as_slice().map_err(value_error)
}

struct SinglePlan<T: WaveletNum> {
    plan: Arc<dyn Dwt<T>>,
    wavelet: String,
    mode: String,
}

impl<T: WaveletNum> SinglePlan<T> {
    fn new(length: usize, wavelet: &str, mode: &str) -> PyResult<Self> {
        let definition: Wavelet = wavelet.parse().map_err(value_error)?;
        let boundary: Boundary = mode.parse().map_err(value_error)?;
        let plan = DwtPlanner::new()
            .plan_dwt(length, &definition, boundary)
            .map_err(value_error)?;
        Ok(Self {
            plan,
            wavelet: definition.to_string(),
            mode: boundary.to_string(),
        })
    }

    fn validate_signal(&self, signal: &[T]) -> PyResult<()> {
        if signal.len() != self.plan.signal_len() {
            return Err(PyValueError::new_err(format!(
                "signal length {} does not match planned length {}",
                signal.len(),
                self.plan.signal_len()
            )));
        }
        Ok(())
    }

    fn validate_coefficients(&self, approx: &[T], detail: &[T]) -> PyResult<()> {
        let expected = self.plan.coeff_len();
        if approx.len() != expected || detail.len() != expected {
            return Err(PyValueError::new_err(format!(
                "coefficient bands must both have length {expected}, got {} and {}",
                approx.len(),
                detail.len()
            )));
        }
        Ok(())
    }
}

struct MultiPlan<T: WaveletNum> {
    plan: Arc<WavedecPlan<T>>,
    wavelet: String,
    mode: String,
}

impl<T: WaveletNum> MultiPlan<T> {
    fn new(length: usize, wavelet: &str, mode: &str, level: Option<usize>) -> PyResult<Self> {
        let definition: Wavelet = wavelet.parse().map_err(value_error)?;
        let boundary: Boundary = mode.parse().map_err(value_error)?;
        let level = level.map_or(Level::Max, Level::Exact);
        let plan = DwtPlanner::new()
            .plan_wavedec(length, &definition, boundary, level)
            .map_err(value_error)?;
        Ok(Self {
            plan,
            wavelet: definition.to_string(),
            mode: boundary.to_string(),
        })
    }

    fn validate_signal(&self, signal: &[T]) -> PyResult<()> {
        if signal.len() != self.plan.signal_len() {
            return Err(PyValueError::new_err(format!(
                "signal length {} does not match planned length {}",
                signal.len(),
                self.plan.signal_len()
            )));
        }
        Ok(())
    }

    fn forward(&self, signal: &[T]) -> Vec<Vec<T>> {
        let decomposition = self.plan.forward(signal);
        decomposition.bands().map(<[T]>::to_vec).collect()
    }

    fn prepare_decomposition(&self, bands: &[&[T]]) -> PyResult<Decomposition<T>> {
        let expected_bands = self.plan.levels() + 1;
        if bands.len() != expected_bands {
            return Err(PyValueError::new_err(format!(
                "expected {expected_bands} coefficient bands, got {}",
                bands.len()
            )));
        }

        let mut decomposition = self.plan.allocate_decomposition();
        let approx = decomposition.approx_mut();
        if bands[0].len() != approx.len() {
            return Err(PyValueError::new_err(format!(
                "approximation band must have length {}, got {}",
                approx.len(),
                bands[0].len()
            )));
        }
        approx.copy_from_slice(bands[0]);

        for (index, band) in bands[1..].iter().enumerate() {
            let level = self.plan.levels() - index;
            let detail = decomposition.detail_mut(level);
            if band.len() != detail.len() {
                return Err(PyValueError::new_err(format!(
                    "detail band cD_{level} must have length {}, got {}",
                    detail.len(),
                    band.len()
                )));
            }
            detail.copy_from_slice(band);
        }
        Ok(decomposition)
    }
}

macro_rules! single_plan_class {
    ($name:ident, $sample:ty, $dtype:literal) => {
        #[doc = "A reusable fixed-length single-level DWT/IDWT plan."]
        #[pyclass(frozen, module = "wavelets_rs._wavelets_rs")]
        struct $name {
            inner: SinglePlan<$sample>,
        }

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (length, wavelet = "db1", mode = "symmetric"))]
            fn new(length: usize, wavelet: &str, mode: &str) -> PyResult<Self> {
                Ok(Self {
                    inner: SinglePlan::new(length, wavelet, mode)?,
                })
            }

            #[getter]
            fn signal_len(&self) -> usize {
                self.inner.plan.signal_len()
            }

            #[getter]
            fn coeff_len(&self) -> usize {
                self.inner.plan.coeff_len()
            }

            #[getter]
            fn wavelet(&self) -> &str {
                &self.inner.wavelet
            }

            #[getter]
            fn mode(&self) -> &str {
                &self.inner.mode
            }

            #[getter]
            fn dtype(&self) -> &'static str {
                $dtype
            }

            fn forward<'py>(
                &self,
                py: Python<'py>,
                signal: PyReadonlyArray1<'py, $sample>,
            ) -> PyResult<(Bound<'py, PyArray1<$sample>>, Bound<'py, PyArray1<$sample>>)> {
                let signal = contiguous_slice(&signal)?;
                self.inner.validate_signal(signal)?;
                let (approx, detail) = py.detach(|| self.inner.plan.forward(signal));
                Ok((approx.into_pyarray(py), detail.into_pyarray(py)))
            }

            fn inverse<'py>(
                &self,
                py: Python<'py>,
                approx: PyReadonlyArray1<'py, $sample>,
                detail: PyReadonlyArray1<'py, $sample>,
            ) -> PyResult<Bound<'py, PyArray1<$sample>>> {
                let approx = contiguous_slice(&approx)?;
                let detail = contiguous_slice(&detail)?;
                self.inner.validate_coefficients(approx, detail)?;
                let output = py.detach(|| self.inner.plan.inverse(approx, detail));
                Ok(output.into_pyarray(py))
            }

            fn __repr__(&self) -> String {
                format!(
                    "{}(length={}, wavelet={:?}, mode={:?})",
                    stringify!($name),
                    self.inner.plan.signal_len(),
                    self.inner.wavelet,
                    self.inner.mode
                )
            }
        }
    };
}

macro_rules! multilevel_plan_class {
    ($name:ident, $sample:ty, $dtype:literal) => {
        #[doc = "A reusable fixed-length multilevel wavedec/waverec plan."]
        #[pyclass(frozen, module = "wavelets_rs._wavelets_rs")]
        struct $name {
            inner: MultiPlan<$sample>,
        }

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (length, wavelet = "db1", mode = "symmetric", level = None))]
            fn new(
                length: usize,
                wavelet: &str,
                mode: &str,
                level: Option<usize>,
            ) -> PyResult<Self> {
                Ok(Self {
                    inner: MultiPlan::new(length, wavelet, mode, level)?,
                })
            }

            #[getter]
            fn signal_len(&self) -> usize {
                self.inner.plan.signal_len()
            }

            #[getter]
            fn levels(&self) -> usize {
                self.inner.plan.levels()
            }

            #[getter]
            fn wavelet(&self) -> &str {
                &self.inner.wavelet
            }

            #[getter]
            fn mode(&self) -> &str {
                &self.inner.mode
            }

            #[getter]
            fn dtype(&self) -> &'static str {
                $dtype
            }

            fn forward<'py>(
                &self,
                py: Python<'py>,
                signal: PyReadonlyArray1<'py, $sample>,
            ) -> PyResult<Vec<Bound<'py, PyArray1<$sample>>>> {
                let signal = contiguous_slice(&signal)?;
                self.inner.validate_signal(signal)?;
                let bands = py.detach(|| self.inner.forward(signal));
                Ok(bands
                    .into_iter()
                    .map(|band| band.into_pyarray(py))
                    .collect())
            }

            fn inverse<'py>(
                &self,
                py: Python<'py>,
                bands: Vec<PyReadonlyArray1<'py, $sample>>,
            ) -> PyResult<Bound<'py, PyArray1<$sample>>> {
                let slices: Vec<_> = bands
                    .iter()
                    .map(contiguous_slice)
                    .collect::<PyResult<_>>()?;
                let decomposition = self.inner.prepare_decomposition(&slices)?;
                let output = py.detach(|| self.inner.plan.inverse(&decomposition));
                Ok(output.into_pyarray(py))
            }

            fn __repr__(&self) -> String {
                format!(
                    "{}(length={}, wavelet={:?}, mode={:?}, level={})",
                    stringify!($name),
                    self.inner.plan.signal_len(),
                    self.inner.wavelet,
                    self.inner.mode,
                    self.inner.plan.levels()
                )
            }
        }
    };
}

single_plan_class!(DwtPlanF32, f32, "float32");
single_plan_class!(DwtPlanF64, f64, "float64");
multilevel_plan_class!(WavedecPlanF32, f32, "float32");
multilevel_plan_class!(WavedecPlanF64, f64, "float64");

#[pymodule]
/// Safe NumPy bindings for the Rust wavelets crate.
fn _wavelets_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<DwtPlanF32>()?;
    module.add_class::<DwtPlanF64>()?;
    module.add_class::<WavedecPlanF32>()?;
    module.add_class::<WavedecPlanF64>()?;
    Ok(())
}
