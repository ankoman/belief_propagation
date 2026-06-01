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
  - ML-DSA-44 (n=256, tau=39): realistic parameters; feasible with tight sigma
    because the adaptive prior reduces inner-loop work to O(sigma^2) per cell.
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
    radius: int,
) -> Dict[int, float]:
    """
    Gaussian prior centred on `measured`, clipped to [x_min, x_max].
    Only values within `radius` of round(measured) are included so the
    support stays small even when the theoretical range is large.
    """
    center = int(round(measured))
    lo = max(x_min, center - radius)
    hi = min(x_max, center + radius)
    raw = {v: math.exp(-((measured - v) ** 2) / (2 * sigma ** 2))
           for v in range(lo, hi + 1)}
    total = sum(raw.values())
    if total == 0:
        return {center: 1.0}
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
    seed: int = 42,
    report_every: int = 10,
) -> Tuple[float, float]:
    """Run the SASCA and return (accuracy, elapsed_seconds)."""
    rng = random.Random(seed)
    secret = random_secret(n, eta, rng)

    bp = MLDsaBP(n, eta)

    x_min = -tau * eta
    x_max =  tau * eta
    # Radius keeps the support small: capture ~6σ so virtually no probability
    # mass is truncated.
    radius = max(1, int(math.ceil(3 * sigma)))

    t0 = time.perf_counter()
    for t in range(num_traces):
        challenge = random_challenge(n, tau, rng)
        x_true    = poly_mul_mod(challenge, secret)

        x_priors = [
            gaussian_prior(xi + rng.gauss(0, sigma), sigma, x_min, x_max, radius)
            for xi in x_true
        ]

        bp.add_trace(challenge, x_priors)

        if (t + 1) % report_every == 0:
            est = bp.get_map_estimate()
            ok  = sum(e == s for e, s in zip(est, secret))
            elapsed = time.perf_counter() - t0
            print(f"    traces={t+1:4d}  correct={ok}/{n} ({100*ok/n:.1f}%)"
                  f"  [{elapsed:.1f}s]")

    est     = bp.get_map_estimate()
    ok      = sum(e == s for e, s in zip(est, secret))
    elapsed = time.perf_counter() - t0
    return ok / n, elapsed


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    configs = [
        # (label,           n,   eta, tau, traces, sigma, report_every)
        ("Demo n=32",       32,  2,   5,   30,     0.5,   5),
        ("ML-DSA-44 n=256", 256, 2,   39,  20,     0.5,   5),
    ]

    for label, n, eta, tau, num_traces, sigma, every in configs:
        print(f"\n=== {label}  (eta={eta}, tau={tau}, traces={num_traces}, sigma={sigma}) ===")
        acc, elapsed = run_attack(n, eta, tau, num_traces, sigma, report_every=every)
        print(f"  => final accuracy {100*acc:.1f}%  total {elapsed:.1f}s")
