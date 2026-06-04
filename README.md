A generic belief propagation implementation in Rust
=======

Initially created for the Chosen Ciphertext k-Trace Attacks on Masked CCA2 Secure Kyber paper (see https://eprint.iacr.org/2021/956.pdf) by Julius Hermelink, Silvan Streit, and Emanuele Strieder.

Below are modified by Junichi Sakamoto (under the sseupport with Claude Code).

## ML-DSA BP (`MLDsaBP`)

`MLDsaBP` is a belief propagation engine for recovering the secret key polynomial
of ML-DSA (FIPS 204) from side-channel leakage.  The factor graph models the
negacyclic convolution constraint `x = c * s` (or `x = c * (t₀ - s)`, etc.)
between the leakage measurements `x` and the unknown polynomial `s`.

### Factor graph structure

Each polynomial coefficient `x_i` becomes a factor node connected to the
`τ` (tau) variables `s_j` for which the challenge weight `c_{i,j} ≠ 0`.
The constraint is:

```
x_i = Σ_j  challenge_weight(c, i, j) · s_j
```

where `challenge_weight` implements the standard negacyclic convolution
(coefficients in `{−1, 0, +1}`).

### Message passing (convaddrev algorithm)

Each BP iteration computes factor→variable messages via a left/right prefix
convolution scheme:

1. **Forward pass** — build `left[k]`, the distribution of
   `Σ_{l<k} c_l · s_{j_l}`, by successive `convadd` calls.
2. **Backward pass** — accumulate `rx = RX_k` where
   `RX_k(u) = Σ_r right[k](r) · msg_x(u + r)`.  
   `RX` starts at `msg_x` and is updated via `convaddrev`
   (= `convadd` with negated weight `−c`), folding `msg_x` in from the right.
3. **Message** — the factor→variable message for `s_{j_k}` is

   ```
   m_k(sv) = Σ_v  left[k](v) · RX_k(v + c_k · sv)
   ```

   This is a dot product of `left[k].data` with a slice of `rx.data`,
   replacing the O(|left| × |right|) triple sum of a naïve implementation.

### Dense array representation

All distributions are stored as `DenseMsg { offset: i32, data: Vec<f64> }` —
a contiguous `f64` array with an integer offset — rather than `HashMap<i32, f64>`.
This eliminates hash overhead and lets LLVM auto-vectorise the inner SAXPY loops.

### Adaptive FFT / direct switching

The `convadd` and cross-correlation kernels dispatch between two backends:

| Condition | Backend | Cost |
|-----------|---------|------|
| `g.len() × n_cav ≤ FFT_THRESHOLD` | **Direct SAXPY** | O(L × K) |
| `g.len() × n_cav > FFT_THRESHOLD` | **FFT convolution** | O(N log N) |

`FFT_THRESHOLD = 65 536` (tunable constant in `ml_dsa_bp.rs`).

The crossover point where FFT becomes faster depends on the kernel size
`n_cav = 2η + 1`:

| η (eta) | n_cav | FFT kicks in at left.len > | Used for ML-DSA? |
|---------|-------|---------------------------|-----------------|
| 2 | 5 | 13 107 | **Never** (max left = 153) |
| 4 | 9 | 7 281 | **Never** (max left = 385) |
| 32 | 65 | 1 008 | Partial (k ≥ 16 of 38) |
| 128 | 257 | 255 | Almost always (k ≥ 1) |
| 4 096 | 8 193 | 8 | Always |

**Consequence**: standard ML-DSA parameters (η = 2 or 4) always use the direct
path with zero FFT overhead.  The FFT path activates automatically for larger η,
such as the `x = c(t₀ − s)` attack formulation where η ≈ 2^(d−1) ≈ 4096.

### Performance overview (n = 256, t = 39, 8 cores)

| Problem | η | Backend | Time / iter |
|---------|---|---------|-------------|
| ML-DSA-44 | 2 | Direct | ~8 ms |
| ML-DSA-65 | 4 | Direct | ~18 ms |
| ML-DSA-87 | 2 | Direct | ~13 ms |
| t₀ − s attack | 4 096 | FFT | ~100 s |

Scaling law (empirical): time ∝ η^2 · t^2 for the direct path;
the FFT path reduces this to roughly O(η · t · N log N) where N ≈ 2 η t.

### `x_priors` and the observable range

`x_priors[i]` is a `dict[int, float]` mapping each possible value of `x_i`
to its prior probability (not log-prob).  The keys must lie within the range
`[−τη, +τη]` that `x_i = (c * s)_i` can actually take, otherwise the factor
messages are identically zero (the dot product finds no overlap).

For the `x = c(t₀ − s)` formulation the keys are integers in
`[−τ(2^(d−1) + η), +τ(2^(d−1) + η)]`.  If the leakage pipeline produces
modular residues in `[0, q)`, centre them first:

```python
q = 8_380_417          # ML-DSA modulus
tau_eta = tau * eta    # e.g. 39 * 2 = 78 for ML-DSA-44

def center_mod(v, bound):
    """Map v ∈ [0, q) to the natural range [−bound, +bound]."""
    return v if v <= bound else v - q
```

## Build

### Rust library

```bash
cargo build --release
```

### Python bindings

Python bindings are built with [maturin](https://github.com/PyO3/maturin).

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install maturin and build the extension module in-place
pip install maturin
maturin develop --release
```

After `maturin develop`, the `belief_propagation` module is importable from within the `.venv`.

### Run the ML-DSA attack example

```bash
source .venv/bin/activate
python examples/ml_dsa_attack.py
```
