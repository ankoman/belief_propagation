import math
import random
import time
import sys, pickle
import numpy as np
import click
from typing import Dict, List, Tuple
from ml_dsa_attack import count_recovered
try:
    from belief_propagation import MLDsaBP, gen_x_priors_parallel
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

def flip_bits_nbit(x, p_bit_error, n_bits):
    y = x & ((1 << n_bits) - 1)  # nbitに制限
    for i in range(n_bits):
        if random.random() < p_bit_error:
            y ^= (1 << i)
    return y
    
def hw(x):
    return bin(x).count("1")

def gen_x_priors(w0_obs, xD_i, x_min, x_max, Azct1_low_i, h_i, B, C, beta, p_bit_error, n_bits, USE_HINT = False) -> Dict[int, float]:
    if USE_HINT:
        x_min_t = -999999999
        x_max_t =  999999999
        if h_i == 0:
            x_min_t = -beta - B - Azct1_low_i
            x_max_t =  beta + B - Azct1_low_i
        elif Azct1_low_i > 0:
            x_min_t = -beta + C - Azct1_low_i
        else:
            x_max_t = beta - C - Azct1_low_i

        x_min = max(x_min, x_min_t)
        x_max = min(x_max, x_max_t) 

    dict_t = {}
    for x_est in range(x_min, x_max + 1):
        e = w0_obs ^ (xD_i + x_est)
        hd = hw(e & ((1 << n_bits) - 1))
        dict_t[x_est] = (1-p_bit_error)**hd
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
    damping = 0.0,
    use_hint: bool = False,
) -> Tuple[float, float]:

    rng = random.Random(seed)
    attack_idx = 0
    t_start = time.perf_counter()
    bp = MLDsaBP(n, eta)
    bp.set_damping(damping)  # loopy BP stabilization for wide secret range. 0 is no effect

    if t0_is_known:
        s_min = -eta
        s_max =  eta
        correct_secret = s2[attack_idx]
        n_bits = 8
    else:
        s_min = -4098
        s_max = 4097
        correct_secret = s2[attack_idx] - t0[attack_idx]
        correct_secret.mod_pm()
        n_bits = 18

    x_min =  tau * s_min
    x_max =  tau * s_max
    p_unif = 1.0 / (s_max - s_min + 1)
    bp.set_prior([{v: p_unif for v in range(s_min, s_max + 1)} for _ in range(n)])
    B = 95232 - tau*eta - 1
    C = 95232 + tau*eta + 1

    # Phase 1: add traces with noisy observations
    for w0, c, xD, Azct1_low, h in list_traces:
        c.mod_pm()
        xD[attack_idx].mod_pm()
        x_priors = []
        # ### Serial version                                                                             
        # for w0_true_i, xD_i, Azct1_low_i, h_i in zip(w0[attack_idx], xD[attack_idx], Azct1_low[attack_idx], h[attack_idx]):
        #     if p_bit_error == 0.0:
        #         x_priors.append({w0_true_i - xD_i: 1.0}) 
        #     else:
        #         w0_obs = flip_bits_nbit(w0_true_i, p_bit_error, n_bits)
        #         dict_t = gen_x_priors(w0_obs, xD_i, x_min, x_max, Azct1_low_i, h_i, B, C, tau*eta, p_bit_error, n_bits)
        #         x_priors.append(dict_t)
        ### Parallel version
        if p_bit_error == 0.0:
            x_priors = [{w0_true_i - xD_i: 1.0} for w0_true_i, xD_i in zip(w0[attack_idx], xD[attack_idx])]
        else:
            w0_obs_list = [flip_bits_nbit(w0_true_i, p_bit_error, n_bits) for w0_true_i in w0[attack_idx].coeff]
            x_priors = gen_x_priors_parallel(
                w0_obs_list,
                list(xD[attack_idx].coeff),
                x_min, x_max,
                list(Azct1_low[attack_idx].coeff) if use_hint else [],
                list(h[attack_idx].coeff) if use_hint else [],
                B, C, tau * eta,
                p_bit_error, n_bits,
                use_hint
            )
        bp.add_trace(c, x_priors)
    print(f"  collected {bp.trace_count()} traces  [{time.perf_counter()-t_start:.1f}s]")

    # Phase 2: iterate BP
    for it in range(1, num_iterations + 1):
        print(f"  iter={it} ",end="")
        bp.run_iteration()
        est      = bp.get_map_estimate()
        lp       = bp.get_log_key_probs()
        ok       = sum(e == s for e, s in zip(est, correct_secret))
        rec      = count_recovered(est, correct_secret, lp)
        maximum = max(abs(e - s) for e,s in zip(est, correct_secret))
        elapsed  = time.perf_counter() - t_start
        print(f"  - Corr={ok}/{n} ({100*ok/n:.1f}%)  Recov={rec}/{n}  [{elapsed:.1f}s]")
        if ok == 256:
            print("Attack success: all coefficients recovered, stopping early.")
            break

    # est     = bp.get_map_estimate()
    # lp      = bp.get_log_key_probs()
    # ok      = sum(e == s for e, s in zip(est, correct_secret))
    # rec     = count_recovered(est, correct_secret, lp)
    # elapsed = time.perf_counter() - t_start
    return ok, rec, n, elapsed
    

@click.command()
@click.option("--p-bit-error", default=0.0,  show_default=True, type=float, help="Bit-flip error rate for observations.")
@click.option("--num-traces",  default=50,    show_default=True, type=int,   help="Number of traces to use.")
@click.option("--num-iter",    default=50,    show_default=True, type=int,   help="Maximum BP iterations.")
@click.option("--damping",     default=0.0,   show_default=True, type=float, help="Message damping factor (0=none, 0.5=recommended for t0-unknown).")
@click.option("--t0-known",    is_flag=True,  default=False,                 help="Use t0-known mode (default: t0-unknown).")
@click.option("--use-hint",    is_flag=True,  default=False,                 help="Use hint-bit constraint (default: no).")
@click.option("--traceset",    default=0,     type=int,                      help="Number of traceset")
def main(p_bit_error, num_traces, num_iter, damping, t0_known, use_hint, traceset):
    n, eta, tau = 256, 2, 39
    t0_is_known = t0_known

    if t0_is_known:
        trace_file = f"traces/t0_known/traces_t0_known_1000_{traceset}.pkl"
    else:
        trace_file = f"traces/t0_unknown/traces_t0_unknown_1000_{traceset}.pkl"

    list_traces = []
    with open(trace_file, "rb") as f:
        t0 = pickle.load(f)
        s2 = pickle.load(f)
        for i in range(num_traces):
            w0 = pickle.load(f)
            c  = pickle.load(f)
            xD = pickle.load(f)
            if t0_is_known == False:
                Azct1_low = pickle.load(f)
                h = pickle.load(f)
                list_traces.append((w0, c, xD, Azct1_low, h))
            else:
                list_traces.append((w0, c, xD, None, None))

    label = f"ML-DSA-44 n={n} ({'t0-known' if t0_is_known else 't0-unknown'})"
    print(f"\n=== {label}  (eta={eta}, tau={tau}, traces={num_traces}, p_bit_error={p_bit_error}, damping={damping}, use_hint={use_hint}) ===")
    ok, rec, n_, elapsed = run_attack(
        n, eta, tau, p_bit_error, list_traces, s2, w0, t0,
        num_iterations=num_iter,t0_is_known=t0_is_known, damping=damping, use_hint=use_hint
    )
    print(f"  => correct={ok}/{n_} ({100*ok/n_:.1f}%)  recovered={rec}/{n_}  total {elapsed:.1f}s")


if __name__ == "__main__":
    main()

