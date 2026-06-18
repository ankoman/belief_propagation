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
import numpy as np

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


def hamming_weight_32(v: int) -> int:
    """Number of 1-bits in the 32-bit two's complement representation."""
    return bin(v & 0xFFFFFFFF).count('1')


def var_hw32_theoretical(tau: int, eta: int) -> float:
    """Theoretical Var(HW32(x_true[i])) for x_true[i] = sum of tau i.i.d. Uniform{-eta..eta}."""
    # Exact distribution by convolution
    dist: Dict[int, float] = {0: 1.0}
    p_single = 1.0 / (2 * eta + 1)
    for _ in range(tau):
        new: Dict[int, float] = {}
        for v1, p1 in dist.items():
            for dv in range(-eta, eta + 1):
                key = v1 + dv
                new[key] = new.get(key, 0.0) + p1 * p_single
        dist = new
    e1 = sum(p * hamming_weight_32(v) for v, p in dist.items())
    e2 = sum(p * hamming_weight_32(v) ** 2 for v, p in dist.items())
    return e2 - e1 ** 2


def _entropy(log_prob: Dict[int, float]) -> float:
    """Shannon entropy of a belief given as {value: log_prob}."""
    vals = list(log_prob.values())
    max_v = max(vals)
    exp_v = [math.exp(v - max_v) for v in vals]
    total = sum(exp_v)
    probs = [e / total for e in exp_v]
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def count_recovered(
    est: List[int],
    secret: List[int],
    log_probs: List[Dict[int, float]],
) -> int:
    """Largest K s.t. top-K most confident (lowest entropy) MAP estimates are all correct."""
    entropies = [_entropy(lp) for lp in log_probs]
    print(f"Ave ent: {sum(entropies)/len(entropies):.5f}",end=', ')
    print(f'Ave dist: {np.abs(np.array(est)-np.array(secret.coeff)).mean():.5f}',end=', ')
    print(f'Max dist: {max(abs(e - s) for e,s in zip(est, secret.coeff))}',end=', ')
    order = sorted(range(len(entropies)), key=lambda i: entropies[i])
    count = 0
    for idx in order:
        if est[idx] == secret[idx]:
            count += 1
        else:
            break
    return count


def observation_likelihood(
    l_obs: float,
    sigma: float,
    x_min: int,
    x_max: int,
) -> Dict[int, float]:
    """P(l_obs | x=v) for each v in [x_min, x_max].

    Leakage model (paper Eq. 12-14): l_obs = HW32(x) + N(0, sigma^2).
    Template: exp(-(l_obs - HW32(v))^2 / (2*sigma^2)).
    """
    raw = {v: math.exp(-((l_obs - hamming_weight_32(v)) ** 2) / (2 * sigma ** 2))
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
    snr: float,
    num_iterations: int = 5,
    seed: int = 10,
) -> Tuple[float, float]:
    """Run the SASCA and return (accuracy, elapsed_seconds)."""
    # Derive sigma from SNR using theoretical HW32 variance
    var_hw = var_hw32_theoretical(tau, eta)
    sigma  = math.sqrt(var_hw / snr)
    print(f"  HW32 var(theoretical)={var_hw:.2f}  sigma={sigma:.4f}")
    
    rng = random.Random(seed)
    secret = random_secret(n, eta, rng)

    x_min = -tau * eta
    x_max =  tau * eta

    t0 = time.perf_counter()

    # Phase 1: generate all (challenge, x_true) pairs
    traces_raw = []
    for _ in range(num_traces):
        challenge = random_challenge(n, tau, rng)
        x_true    = poly_mul_mod(challenge, secret)
        traces_raw.append((challenge, x_true))

    # Phase 2: add traces with noisy observations
    bp = MLDsaBP(n, eta)
    p_unif = 1.0 / (2 * eta + 1)
    bp.set_prior([{v: p_unif for v in range(-eta, eta + 1)} for _ in range(n)])
    for challenge, x_true in traces_raw:
        x_obs    = [hamming_weight_32(xi) + rng.gauss(0, sigma) for xi in x_true]
        x_priors = [observation_likelihood(obs, sigma, x_min, x_max) for obs in x_obs]
        bp.add_trace(challenge, x_priors)
    print(f"  collected {bp.trace_count()} traces  [{time.perf_counter()-t0:.1f}s]")

    # Phase 2: iterate BP
    for it in range(1, num_iterations + 1):
        bp.run_iteration()
        est      = bp.get_map_estimate()
        lp       = bp.get_log_key_probs()
        ok       = sum(e == s for e, s in zip(est, secret))
        rec      = count_recovered(est, secret, lp)
        elapsed  = time.perf_counter() - t0
        print(f"  iter={it}  correct={ok}/{n} ({100*ok/n:.1f}%)  recovered={rec}/{n}  [{elapsed:.1f}s]")

    est     = bp.get_map_estimate()
    lp      = bp.get_log_key_probs()
    ok      = sum(e == s for e, s in zip(est, secret))
    rec     = count_recovered(est, secret, lp)
    elapsed = time.perf_counter() - t0
    return ok, rec, n, elapsed


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    configs = [
        # (label,           n,   eta, tau, traces, snr,  iters)
        # ("Demo n=32",       32,  2,   5,   20,     10.0, 5),
        ("ML-DSA-44 n=256", 256, 2,   39,  22,     0.1, 20),
    ]

    for label, n, eta, tau, num_traces, snr, iters in configs:
        print(f"\n=== {label}  (eta={eta}, tau={tau}, traces={num_traces}, snr={snr}) ===")
        ok, rec, n_, elapsed = run_attack(n, eta, tau, num_traces, snr, num_iterations=iters)
        print(f"  => correct={ok}/{n_} ({100*ok/n_:.1f}%)  recovered={rec}/{n_}  total {elapsed:.1f}s")
