import math
import random
import time
import sys, pickle
import numpy as np
from typing import Dict, List, Tuple
from ml_dsa_attack import count_recovered
try:
    from belief_propagation import MLDsaBP
except ImportError:
    sys.exit(
        "belief_propagation not found – run `maturin develop` inside .venv first."
    )

class polyRing:
    q = 8380417
    n = 256

    def __init__(self):
        self.coeff = [0] * polyRing.n

    def __repr__(self):
        # return str(list(map(hex, self.coeff)))
        return str(self.coeff)
    
    def __getitem__(self, index):
        return self.coeff[index]
    
    def __neg__(self):
        tmp = self.__class__()
        for i in range(self.n):
            tmp.coeff[i] = -self.coeff[i] % self.q
        return tmp
    
    def __add__(self, other):
        tmp = self.__class__()
        for i in range(self.n):
            tmp.coeff[i] = (self.coeff[i] + other.coeff[i]) % self.q ### reduction
        return tmp
    
    def __mul__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        tmp = self.__class__()
        for i in range(self.n):
            tmp.coeff[i] = (self.coeff[i] * other) % self.q ### reduction
        return tmp
    
    def __sub__(self, other):
        tmp = self.__class__()
        for i in range(self.n):
            tmp.coeff[i] = (self.coeff[i] - other.coeff[i]) % self.q ### reduction
        return tmp

    def __lshift__(self, shift: int):
        tmp = self.__class__()
        for i in range(self.n):
            tmp.coeff[i] = self.coeff[i] << shift
        return tmp
    
    def mod_pm(self):
        for i in range(self.n):
            if self.coeff[i] > self.q // 2:
                self.coeff[i] -= self.q

def run_attack(
    n: int,
    eta: int,
    tau: int,
    snr: float,
    list_traces,
    s2,
    w0,
    num_iterations: int = 5,
    seed: int = 10,
) -> Tuple[float, float]:

    rng = random.Random(seed)

    x_min = -tau * eta
    x_max =  tau * eta

    t0 = time.perf_counter()
    ### Attack only s2[0]
    attack_idx = 0

    # Phase 2: add traces with noisy observations
    bp = MLDsaBP(n, eta)
    p_unif = 1.0 / (2 * eta + 1)
    bp.set_prior([{v: p_unif for v in range(-eta, eta + 1)} for _ in range(n)])
    for w0, c, x_D in list_traces:
        c.mod_pm()
        x = np.array(w0) - np.array(x_D)
        x[attack_idx].mod_pm()
        x_priors = []
        for xi in x[attack_idx]:
            dict_t = {v: 0.0 for v in range(x_min, x_max + 1)}
            dict_t[xi] = 1.0
            x_priors.append(dict_t)
        bp.add_trace(c, x_priors)
    print(f"  collected {bp.trace_count()} traces  [{time.perf_counter()-t0:.1f}s]")

    # Phase 2: iterate BP
    for it in range(1, num_iterations + 1):
        bp.run_iteration()
        est      = bp.get_map_estimate()
        lp       = bp.get_log_key_probs()
        ok       = sum(e == s for e, s in zip(est, s2[attack_idx]))
        rec      = count_recovered(est, s2[attack_idx], lp)
        elapsed  = time.perf_counter() - t0
        print(f"  iter={it}  correct={ok}/{n} ({100*ok/n:.1f}%)  recovered={rec}/{n}  [{elapsed:.1f}s]")

    est     = bp.get_map_estimate()
    lp      = bp.get_log_key_probs()
    ok      = sum(e == s for e, s in zip(est, s2[attack_idx]))
    rec     = count_recovered(est, s2[attack_idx], lp)
    elapsed = time.perf_counter() - t0
    return ok, rec, n, elapsed


if __name__ == "__main__":
    configs = [
        ("ML-DSA-44 n=256", 256, 2,   39,  0.01, 20),
    ]

    list_traces = []
    with open("traces_t0_known_100.pkl", "rb") as f:
        t0 = pickle.load(f)
        s2 = pickle.load(f)
        for _ in range(3):
            w0 = pickle.load(f)
            c = pickle.load(f)
            x_D = pickle.load(f)
            list_traces.append((w0, c, x_D))

    for label, n, eta, tau, snr, iters in configs:
        print(f"\n=== {label}  (eta={eta}, tau={tau}, snr={snr}) ===")
        ok, rec, n_, elapsed = run_attack(n, eta, tau, snr, list_traces, s2, w0, num_iterations=iters)
        print(f"  => correct={ok}/{n_} ({100*ok/n_:.1f}%)  recovered={rec}/{n_}  total {elapsed:.1f}s")

    # for label, n, eta, tau, num_traces, snr, iters in configs:
    #     print(f"\n=== {label}  (eta={eta}, tau={tau}, traces={num_traces}, snr={snr}) ===")
    #     ok, rec, n_, elapsed = run_attack(n, eta, tau, num_traces, snr, num_iterations=iters)
    #     print(f"  => correct={ok}/{n_} ({100*ok/n_:.1f}%)  recovered={rec}/{n_}  total {elapsed:.1f}s")
