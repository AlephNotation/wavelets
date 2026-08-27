# wavelets-rs

Python bindings for the Rust [`wavelets`](https://github.com/AlephNotation/wavelets)
crate. The API preserves the core plan-once execution model: construct a plan
for a fixed signal shape, then reuse it for NumPy-in/NumPy-out transforms.

```python
import numpy as np
import wavelets_rs

signal = np.arange(4096, dtype=np.float64)
plan = wavelets_rs.plan_dwt(len(signal), "db4", mode="symmetric")
approx, detail = plan.forward(signal)
reconstructed = plan.inverse(approx, detail)
```

The package is currently developed and benchmarked with the Rust crate. It is
not yet published to PyPI.

Both `float32` and `float64` plans are available for every built-in wavelet and
all nine boundary modes supported by the Rust crate. Inputs must be contiguous,
one-dimensional NumPy arrays of the plan's exact dtype. Transform execution
releases the GIL. NumPy takes ownership of each returned Rust `Vec`, so the
final Rust-to-NumPy conversion does not copy its elements.

Multilevel plans follow PyWavelets coefficient ordering:

```python
plan = wavelets_rs.plan_wavedec(len(signal), "db4", mode="symmetric")
coefficients = plan.forward(signal)  # [cA_L, cD_L, ..., cD_1]
reconstructed = plan.inverse(coefficients)
```

## Development

From the repository root:

```text
python3 -m venv python/.venv
python/.venv/bin/python -m pip install -r python/requirements-dev.txt
(cd python && .venv/bin/maturin develop --release)
python/.venv/bin/python -m pytest python/tests
```

The same-interpreter benchmark and its methodology are documented in the
[repository benchmark guide](https://github.com/AlephNotation/wavelets/tree/master/benchmarks#same-interpreter-python-comparison).
