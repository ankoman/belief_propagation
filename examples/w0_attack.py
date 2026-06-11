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

def flip_bits_7bit(x, p_bit_error):
    y = x & 0x7F  # 7bitに制限
    for i in range(7):
        if random.random() < p_bit_error:
            y ^= (1 << i)
    return y
    
def hw(x):
    return bin(x).count("1")

def gen_x_priors(w0_obs, xD_i, x_min, x_max, p_bit_error):
    dict_t = {}
    for v in range(x_min, x_max + 1):
        hd = hw(((v+xD_i) ^ w0_obs) & 0x7F)
        dict_t[v] = (1-p_bit_error)**hd
    return dict_t

def run_attack(
    n: int,
    eta: int,
    tau: int,
    p_bit_error: float,
    list_traces,
    s2,
    w0,
    t0,
    num_iterations: int = 5,
    seed: int = 10,
    t0_is_known: bool = True,
) -> Tuple[float, float]:

    rng = random.Random(seed)
    attack_idx = 0
    t_start = time.perf_counter()
    bp = MLDsaBP(n, eta)


    if t0_is_known:
        x_min = -tau * eta
        x_max =  tau * eta
        correct_secret = s2[attack_idx]
        p_unif = 1.0 / (2 * eta + 1)
        bp.set_prior([{v: p_unif for v in range(-eta, eta + 1)} for _ in range(n)])
    else:
        x_min = tau * -4098
        x_max = tau * 4097
        correct_secret = s2[attack_idx] - t0[attack_idx] 
        p_unif = 1.0 / (4097+4098)
        bp.set_prior([{v: p_unif for v in range(-4098, 4097 + 1)} for _ in range(n)])


    # Phase 2: add traces with noisy observations
    for w0, c, xD in list_traces:
        c.mod_pm()
        xD[attack_idx].mod_pm()
        x_priors = []
        for w0_true, xD_i in zip(w0[attack_idx], xD[attack_idx]):
            if p_bit_error == 0.0:
                dict_t = {x: 0.0 for x in range(x_min, x_max + 1)}
                dict_t[w0_true - xD_i] = 1.0
                x_priors.append(dict_t)
            else:
                w0_obs = flip_bits_7bit(w0_true, p_bit_error)
                dict_t = gen_x_priors(w0_obs, xD_i, x_min, x_max, p_bit_error)
                x_priors.append(dict_t)
        bp.add_trace(c, x_priors)
    print(f"  collected {bp.trace_count()} traces  [{time.perf_counter()-t_start:.1f}s]")

    # Phase 2: iterate BP
    for it in range(1, num_iterations + 1):
        bp.run_iteration()
        est      = bp.get_map_estimate()
        lp       = bp.get_log_key_probs()
        ok       = sum(e == s for e, s in zip(est, correct_secret))
        print(est)
        print(correct_secret)
        rec      = count_recovered(est, correct_secret, lp)
        elapsed  = time.perf_counter() - t_start
        print(f"  iter={it}  correct={ok}/{n} ({100*ok/n:.1f}%)  recovered={rec}/{n}  [{elapsed:.1f}s]")

    est     = bp.get_map_estimate()
    lp      = bp.get_log_key_probs()
    ok      = sum(e == s for e, s in zip(est, correct_secret))
    rec     = count_recovered(est, correct_secret, lp)
    elapsed = time.perf_counter() - t_start
    return ok, rec, n, elapsed


if __name__ == "__main__":
    configs = [
        ("ML-DSA-44 n=256", 256, 2,   39,  0.0, 10),
    ]

    ### with t0
    # list_traces = []
    # with open("traces_t0_known_100.pkl", "rb") as f:
    #     t0 = pickle.load(f)
    #     s2 = pickle.load(f)
    #     for _ in range(100):
    #         w0 = pickle.load(f)
    #         c = pickle.load(f)
    #         x_D = pickle.load(f)
    #         list_traces.append((w0, c, x_D))

    # for label, n, eta, tau, _, iters in configs:
    #     for p_bit_error in [0.4]:
    #         for num_traces in [70, 80, 90, 100]:
    #             print(f"\n=== {label}  (eta={eta}, tau={tau}, p_bit_error={p_bit_error}) ===")
    #             ok, rec, n_, elapsed = run_attack(n, eta, tau, p_bit_error, list_traces[:num_traces], s2, w0, t0, num_iterations=iters)
    #             print(f"  => correct={ok}/{n_} ({100*ok/n_:.1f}%)  recovered={rec}/{n_}  total {elapsed:.1f}s")

    ### without t0
    list_traces = []
    with open("traces_t0_unknown_100.pkl", "rb") as f:
        t0 = pickle.load(f)
        s2 = pickle.load(f)
        for _ in range(100):
            w0 = pickle.load(f)
            c = pickle.load(f)
            x_D = pickle.load(f)
            list_traces.append((w0, c, x_D))

    for label, n, eta, tau, p_bit_error, num_traces in configs:
        print(f"\n=== {label}  (eta={eta}, tau={tau}, traces={num_traces}, p_bit_error={p_bit_error}) ===")
        ok, rec, n_, elapsed = run_attack(n, eta, tau, p_bit_error, list_traces[:num_traces], s2, w0, t0, num_iterations=10, t0_is_known=False)
        print(f"  => correct={ok}/{n_} ({100*ok/n_:.1f}%)  recovered={rec}/{n_}  total {elapsed:.1f}s")
