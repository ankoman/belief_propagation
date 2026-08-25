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

q = 8380417
SHARES = 4
DELTA = 44
GAMMA_2 = (q-1)/(2*DELTA)
RHO = 25
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

def gen_x_priors(w1, obs_chi, xD_i, x_min, x_max, Azct1_low_i, h_i, U, V, beta, p_bit_error, n_bits, USE_HINT = False) -> Dict[int, float]:
    if USE_HINT:
        x_min_t = -999999999
        x_max_t =  999999999
        if h_i == 0:
            x_min_t = -beta - U - Azct1_low_i
            x_max_t =  beta + U - Azct1_low_i
        elif Azct1_low_i > 0:
            x_min_t = -beta + V - Azct1_low_i
        else:
            x_max_t = beta - V - Azct1_low_i

        x_min = max(x_min, x_min_t)
        x_max = min(x_max, x_max_t) 

    dict_t = {}
    for w0 in range(x_min + xD_i, x_max + xD_i + 1):
        est_chi = math.floor((w0*DELTA-w1)*2**RHO/q+2**(RHO-1))
        hd = hw(est_chi ^ obs_chi)
        dict_t[w0 - xD_i] = (p_bit_error*(1-p_bit_error))**hd
    return dict_t

def obs_SecDecomposeComp(w):
    w = w + q if w < 0 else w
    x = make_share(w, q)
    z = []
    switchedmod = (DELTA << RHO)
    for i in range(SHARES):
        z.append((x[i]*DELTA*2**RHO)//q % switchedmod)
    z[0] = (z[0] + SHARES-1 + 2**(RHO-1)) % switchedmod

    z = unmask(z, switchedmod)

    ### zの下位RHOビットが漏れる
    obs = z & (2**RHO-1)

    return obs

def make_share(x, mod):
    y = [x]
    for i in range(SHARES-1):
        r = random.randint(0, q-1)
        y.append(r)
        y[0] = (y[0] - r) % mod
    return y

def unmask(share, mod):
    x = 0
    for i in range(SHARES):
        x += share[i]
    return x % mod

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
    else:
        s_min = -4097
        s_max = 4098
        correct_secret = s2[attack_idx] - t0[attack_idx]
        correct_secret.mod_pm()

    x_min =  tau * s_min
    x_max =  tau * s_max
    p_unif = 1.0 / (s_max - s_min + 1)
    bp.set_prior([{v: p_unif for v in range(s_min, s_max + 1)} for _ in range(n)])
    U = GAMMA_2 - tau*eta - 1
    V = GAMMA_2 + tau*eta + 1

    # Phase 1: add traces with noisy observations
    for w, w1, w0, c, xD, Azct1_low, h in list_traces:
        c.mod_pm()
        xD[attack_idx].mod_pm()
        x_priors = []
        # ### Serial version                                                                             
        # for w_i, w1_i, w0_true_i, xD_i, Azct1_low_i, h_i in zip(w[attack_idx], w1[attack_idx], w0[attack_idx], xD[attack_idx], Azct1_low[attack_idx], h[attack_idx]):
        #     chi = obs_SecDecomposeComp(w_i)
        #     if p_bit_error == 0.0:
        #         x_priors.append({w0_true_i - xD_i: 1.0}) 
        #     else:
        #         obs_chi = flip_bits_nbit(chi, p_bit_error, RHO)
        #         dict_t = gen_x_priors(w1_i, obs_chi, xD_i, x_min, x_max, Azct1_low_i, h_i, U, V, tau*eta, p_bit_error, RHO)
        #         x_priors.append(dict_t)
        ## Parallel version
        if p_bit_error == 0.0:
            x_priors = [{w0_true_i - xD_i: 1.0} for w0_true_i, xD_i in zip(w0[attack_idx], xD[attack_idx])]
        else:
            obs_chi_list = [flip_bits_nbit(obs_SecDecomposeComp(w_i), p_bit_error, RHO) for w_i in w[attack_idx].coeff]
            x_priors = gen_x_priors_parallel(
                list(w1[attack_idx].coeff),
                obs_chi_list,
                list(xD[attack_idx].coeff),
                x_min, x_max,
                list(Azct1_low[attack_idx].coeff) if use_hint else [],
                list(h[attack_idx].coeff) if use_hint else [],
                U, V, tau * eta,
                p_bit_error,
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
            w = pickle.load(f)
            w1 = pickle.load(f)
            w0 = pickle.load(f)
            c  = pickle.load(f)
            xD = pickle.load(f)
            if t0_is_known == False:
                Azct1_low = pickle.load(f)
                h = pickle.load(f)
                list_traces.append((w, w1, w0, c, xD, Azct1_low, h))
            else:
                list_traces.append((w, w1, w0, c, xD, xD, xD))

    label = f"ML-DSA-44 n={n} ({'t0-known' if t0_is_known else 't0-unknown'})"
    print(f"\n=== {label}  (eta={eta}, tau={tau}, traces={num_traces}, p_bit_error={p_bit_error}, damping={damping}, use_hint={use_hint}) ===")
    ok, rec, n_, elapsed = run_attack(
        n, eta, tau, p_bit_error, list_traces, s2, w0, t0,
        num_iterations=num_iter,t0_is_known=t0_is_known, damping=damping, use_hint=use_hint
    )
    print(f"  => correct={ok}/{n_} ({100*ok/n_:.1f}%)  recovered={rec}/{n_}  total {elapsed:.1f}s")


if __name__ == "__main__":
    main()

