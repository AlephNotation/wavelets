use std::collections::HashMap;
use std::sync::{Arc, Weak};

use fearless_simd::Level as SimdLevel;

use crate::decomposition::{Level, WavedecPlan, resolve_levels};
use crate::plan::{Dwt, PreparedFilterBank, create_dwt_plan};
use crate::{Boundary, Wavelet, WaveletError, WaveletNum};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PlanKey {
    signal_len: usize,
    wavelet_id: u64,
    boundary: Boundary,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct MultilevelPlanKey {
    signal_len: usize,
    wavelet_id: u64,
    boundary: Boundary,
    levels: usize,
}

/// Creates and caches fixed-length discrete wavelet transform plans.
///
/// The planner detects the best available safe SIMD backend once. Repeated
/// requests using the same live [`Wavelet`] and transform configuration share
/// the cached plan.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use wavelets::{Boundary, DwtPlanner, Wavelet};
///
/// let wavelet = Wavelet::haar();
/// let mut planner = DwtPlanner::<f64>::new();
/// let first = planner.plan_dwt(128, &wavelet, Boundary::Periodization)?;
/// let second = planner.plan_dwt(128, &wavelet, Boundary::Periodization)?;
/// assert!(Arc::ptr_eq(&first, &second));
/// # Ok::<(), wavelets::WaveletError>(())
/// ```
pub struct DwtPlanner<T: WaveletNum> {
    cache: HashMap<PlanKey, Weak<dyn Dwt<T>>>,
    multilevel_cache: HashMap<MultilevelPlanKey, Weak<WavedecPlan<T>>>,
    simd_level: SimdLevel,
}

impl<T: WaveletNum> DwtPlanner<T> {
    /// Constructs an empty planner.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            multilevel_cache: HashMap::new(),
            simd_level: SimdLevel::new(),
        }
    }

    /// Plans a one-level transform for signals of exactly `len` samples.
    ///
    /// Planning validates the boundary/length combination and prepares the
    /// edge-extension, polyphase, and applicable structured-signal filter
    /// layouts. Repeated identical requests reuse the same live plan.
    ///
    /// # Errors
    ///
    /// Returns [`WaveletError::EmptySignal`] for `len == 0`, or
    /// [`WaveletError::BoundaryRequiresLongerSignal`] when the selected
    /// extension mode is undefined for `len`. Returns
    /// [`WaveletError::InvalidFilterBank`] if a filter coefficient or scale
    /// becomes non-finite when converted to `T`.
    pub fn plan_dwt(
        &mut self,
        len: usize,
        wavelet: &Wavelet,
        boundary: Boundary,
    ) -> Result<Arc<dyn Dwt<T>>, WaveletError> {
        let key = PlanKey {
            signal_len: len,
            wavelet_id: wavelet.id(),
            boundary,
        };
        if let Some(plan) = self.cache.get(&key).and_then(Weak::upgrade) {
            return Ok(plan);
        }

        let plan: Arc<dyn Dwt<T>> =
            Arc::new(create_dwt_plan(len, wavelet, boundary, self.simd_level)?);
        self.cache.retain(|_, cached| cached.strong_count() > 0);
        self.cache.insert(key, Arc::downgrade(&plan));
        Ok(plan)
    }

    /// Plans a multilevel transform for signals of exactly `len` samples.
    ///
    /// Every single-level plan, band offset, and scratch region is prepared up
    /// front. Repeated requests resolving to the same number of levels reuse
    /// the same live plan.
    ///
    /// # Errors
    ///
    /// Returns [`WaveletError::EmptySignal`] for `len == 0`,
    /// [`WaveletError::InvalidLevel`] when an exact level exceeds the maximum,
    /// [`WaveletError::InvalidFilterBank`] if a filter coefficient or scale
    /// becomes non-finite when converted to `T`, or a boundary/length planning
    /// error at an intermediate level.
    pub fn plan_wavedec(
        &mut self,
        len: usize,
        wavelet: &Wavelet,
        boundary: Boundary,
        level: Level,
    ) -> Result<Arc<WavedecPlan<T>>, WaveletError> {
        let levels = resolve_levels(len, wavelet.filter_len(), level)?;
        let key = MultilevelPlanKey {
            signal_len: len,
            wavelet_id: wavelet.id(),
            boundary,
            levels,
        };
        if let Some(plan) = self.multilevel_cache.get(&key).and_then(Weak::upgrade) {
            return Ok(plan);
        }

        let filters = PreparedFilterBank::new(wavelet, boundary == Boundary::Periodization)?;
        let plan = Arc::new(WavedecPlan::new(
            len,
            wavelet,
            boundary,
            levels,
            filters,
            self.simd_level,
        )?);
        self.multilevel_cache
            .retain(|_, cached| cached.strong_count() > 0);
        self.multilevel_cache.insert(key, Arc::downgrade(&plan));
        Ok(plan)
    }
}

impl<T: WaveletNum> Default for DwtPlanner<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuses_live_plans() {
        let mut planner = DwtPlanner::<f64>::new();
        let wavelet = Wavelet::haar();
        let first = planner.plan_dwt(8, &wavelet, Boundary::Symmetric).unwrap();
        let second = planner.plan_dwt(8, &wavelet, Boundary::Symmetric).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn reuses_equivalent_live_multilevel_plans() {
        let mut planner = DwtPlanner::<f64>::new();
        let wavelet = Wavelet::haar();
        let first = planner
            .plan_wavedec(16, &wavelet, Boundary::Symmetric, Level::Max)
            .unwrap();
        let second = planner
            .plan_wavedec(16, &wavelet, Boundary::Symmetric, Level::Exact(4))
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn prunes_expired_single_level_plans() {
        let mut planner = DwtPlanner::<f64>::new();
        let wavelet = Wavelet::haar();
        drop(planner.plan_dwt(8, &wavelet, Boundary::Symmetric).unwrap());

        let _live = planner.plan_dwt(16, &wavelet, Boundary::Symmetric).unwrap();
        assert_eq!(planner.cache.len(), 1);
    }

    #[test]
    fn prunes_expired_multilevel_plans() {
        let mut planner = DwtPlanner::<f64>::new();
        let wavelet = Wavelet::haar();
        drop(
            planner
                .plan_wavedec(16, &wavelet, Boundary::Symmetric, Level::Exact(2))
                .unwrap(),
        );

        let _live = planner
            .plan_wavedec(32, &wavelet, Boundary::Symmetric, Level::Exact(2))
            .unwrap();
        assert_eq!(planner.multilevel_cache.len(), 1);
    }
}
