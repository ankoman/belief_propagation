"""
ML-DSA SASCA (Soft Analytical Side-Channel Attack) sample using MLDsaBP.

Scenario
--------
The attacker observes multiple signatures.  For each signature the device
computes  x = c · s  (polynomial product in Z[X]/(X^n + 1))  and leaks
noisy measurements of every coefficient of x via a power/EM side channel.
Given enough traces the attacker recovers the secret-key polynomial s.

Two configurations are shown:
  - Demo (n=32):  fast, finishes in a few seconds.
  - ML-DSA-44 (n=256, tau=39): realistic parameters.
"""

import math
import random
import time
import sys
from typing import Dict, List, Tuple

try:
    from belief_propagation import MLDsaBP
except ImportError:
    sys.exit(
        "belief_propagation not found – run `maturin develop` inside .venv first."
    )


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def random_secret(n: int, eta: int, rng: random.Random) -> List[int]:
    return [rng.randint(-eta, eta) for _ in range(n)]


def random_challenge(n: int, tau: int, rng: random.Random) -> List[int]:
    c = [0] * n
    for p in rng.sample(range(n), tau):
        c[p] = rng.choice([-1, 1])
    return c


def poly_mul_mod(c: List[int], s: List[int]) -> List[int]:
    """Multiply c · s in Z[X]/(X^n + 1)."""
    n = len(c)
    x = [0] * n
    for i in range(n):
        if c[i] == 0:
            continue
        for j in range(n):
            k = i + j
            if k < n:
                x[k] += c[i] * s[j]
            else:
                x[k - n] -= c[i] * s[j]
    return x


def gaussian_prior(
    measured: float,
    sigma: float,
    x_min: int,
    x_max: int,
) -> Dict[int, float]:
    """Gaussian prior over the full support [x_min, x_max]."""
    raw = {v: math.exp(-((measured - v) ** 2) / (2 * sigma ** 2))
           for v in range(x_min, x_max + 1)}
    total = sum(raw.values())
    if total == 0:
        return {(x_min + x_max) // 2: 1.0}
    return {v: p / total for v, p in raw.items()}


# -----------------------------------------------------------------------
# Attack
# -----------------------------------------------------------------------

def run_attack(
    n: int,
    eta: int,
    tau: int,
    num_traces: int,
    sigma: float,
    num_iterations: int = 5,
    seed: int = 42,
) -> Tuple[float, float]:
    """Run the SASCA and return (accuracy, elapsed_seconds)."""
    rng = random.Random(seed)
    secret = random_secret(n, eta, rng)

    bp = MLDsaBP(n, eta)

    x_min = -tau * eta
    x_max =  tau * eta

    t0 = time.perf_counter()

    # Phase 1: collect all traces
    for _ in range(num_traces):
        challenge = random_challenge(n, tau, rng)
        x_true    = poly_mul_mod(challenge, secret)
        x_priors  = [
            gaussian_prior(xi + rng.gauss(0, sigma), sigma, x_min, x_max)
            for xi in x_true
        ]
        bp.add_trace(challenge, x_priors)
    print(f"  collected {bp.trace_count()} traces  [{time.perf_counter()-t0:.1f}s]")

    # Phase 2: iterate BP
    for it in range(1, num_iterations + 1):
        bp.run_iteration()
        est = bp.get_map_estimate()
        ok  = sum(e == s for e, s in zip(est, secret))
        elapsed = time.perf_counter() - t0
        print(f"  iter={it}  correct={ok}/{n} ({100*ok/n:.1f}%)  [{elapsed:.1f}s]")

    est     = bp.get_map_estimate()
    ok      = sum(e == s for e, s in zip(est, secret))
    elapsed = time.perf_counter() - t0
    return ok / n, elapsed


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    configs = [
        # (label,           n,   eta, tau, traces, sigma, iters)
        ("Demo n=32",       32,  2,   5,   20,     0.5,   5),
        ("ML-DSA-44 n=256", 256, 2,   39,  20,     0.5,   5),
    ]

    for label, n, eta, tau, num_traces, sigma, iters in configs:
        print(f"\n=== {label}  (eta={eta}, tau={tau}, traces={num_traces}, sigma={sigma}) ===")
        acc, elapsed = run_attack(n, eta, tau, num_traces, sigma, num_iterations=iters)
        print(f"  => final accuracy {100*acc:.1f}%  total {elapsed:.1f}s")
