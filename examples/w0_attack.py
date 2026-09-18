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
ETA = None
TAU = None
DELTA = None
GAMMA_2 = None
RHO = 25

p_kc = [0.563, 0.604, 0.55, 0.617, 0.55, 0.614, 0.558, 0.616, 0.56, 0.622, 0.561, 0.629, 0.55, 0.634, 0.563, 0.645, 0.571, 0.645, 0.572, 0.642, 0.574, 0.648, 0.57, 0.663, 0.556]
p_dl = [0.69, 0.71, 0.628, 0.71, 0.626, 0.715, 0.627, 0.71, 0.632, 0.718, 0.633, 0.724, 0.635, 0.739, 0.636, 0.735, 0.641, 0.75, 0.649, 0.756, 0.662, 0.773, 0.67, 0.798, 0.785]
p_kc = [1-x for x in p_kc]
p_dl = [1-x for x in p_dl]

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

def flip_bits_nbit(x, p_list, n_bits):
    y = x & ((1 << n_bits) - 1)  # nbitに制限
    for i in range(n_bits):
        if random.random() < p_list[i]:
            y ^= (1 << i)
    return y
    
def hw(x):
    return bin(x).count("1")

def gen_x_priors(w1, obs_chi, xD_i, x_min, x_max, Azct1_low_i, h_i, U, V, beta, p_list, n_bits, USE_HINT = False) -> Dict[int, float]:
    # Reference implementation of the leakage model (the production path uses the
    # fused Rust compute_x_prior_dense). p_list holds the per-bit error rates
    # (RHO values). chi is unsigned RHO-bit, so any est_chi outside [0, 2^RHO)
    # gets probability 0. The two lowest chi bits are dropped (>>2, candidate A:
    # shifted bit j maps to raw chi bit j+2), and each differing bit multiplies
    # its per-bit weight p*(1-p).
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

    two_rho = 1 << RHO
    dict_t = {}
    for w0 in range(x_min + xD_i, x_max + xD_i + 1):
        est_chi = math.floor((w0*DELTA-w1)*2**RHO/q+2**(RHO-1))
        if est_chi < 0 or est_chi >= two_rho:
            dict_t[w0 - xD_i] = 0.0
            continue
        diff = (est_chi ^ obs_chi) >> 2
        weight = 1.0
        j = 0
        while diff:
            if diff & 1:
                weight *= p_list[j + 2] * (1 - p_list[j + 2])
            diff >>= 1
            j += 1
        dict_t[w0 - xD_i] = weight
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
    n = 256
    rng = random.Random(seed)
    attack_idx = 0
    t_start = time.perf_counter()
    bp = MLDsaBP(n, ETA)
    bp.set_damping(damping)  # loopy BP stabilization for wide secret range. 0 is no effect

    if t0_is_known:
        s_min = -ETA
        s_max =  ETA
        correct_secret = s2[attack_idx]
    else:
        # s = s2 - t0 with t0 = t mod± 2^13 ∈ [-2^12, 2^12-1], so the range
        # depends on eta and must not stay hardwired to the eta=2 values.
        s_min = -(ETA + 4095)
        s_max =   ETA + 4096
        correct_secret = s2[attack_idx] - t0[attack_idx]
        correct_secret.mod_pm()

    x_min =  TAU * s_min
    x_max =  TAU * s_max
    p_unif = 1.0 / (s_max - s_min + 1)
    bp.set_prior([{v: p_unif for v in range(s_min, s_max + 1)} for _ in range(n)])
    U = GAMMA_2 - TAU*ETA - 1
    V = GAMMA_2 + TAU*ETA + 1

    # Resolve per-bit error rates (a list of RHO values) used both to inject the
    # observation noise and to build the Rust likelihood. The two sentinels pick
    # the measured per-bit rates; a float means a uniform rate on every bit.
    if p_bit_error == "USE_P_KC":
        p_list = list(p_kc)
    elif p_bit_error == "USE_P_DL":
        p_list = list(p_dl)
    else:
        p_list = [float(p_bit_error)] * RHO

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
            bp.add_trace(list(c.coeff), x_priors)
        else:
            # Fused path: compute x_priors directly inside Rust and store them
            # without ever materialising a Python dict (avoids a large transient
            # allocation per trace).
            obs_chi_list = [flip_bits_nbit(obs_SecDecomposeComp(w_i), p_list, RHO) for w_i in w[attack_idx].coeff]
            bp.add_trace_from_leakage(
                list(c.coeff),
                list(w1[attack_idx].coeff),
                obs_chi_list,
                list(xD[attack_idx].coeff),
                x_min, x_max,
                list(Azct1_low[attack_idx].coeff) if use_hint else [],
                list(h[attack_idx].coeff) if use_hint else [],
                U, V, TAU * ETA, DELTA,
                p_list,
                use_hint
            )
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
@click.option("--level",       default=2,     type=int,                      help="Security category in [2,3,5]")
@click.option("--p-bit-error", default="0.0", show_default=True, type=str, help="Bit-flip error rate for observations. Pass 'USE_P_KC' or 'USE_P_DL' to use the per-bit measured error rates, or a float for a uniform rate.")
@click.option("--num-traces",  default=50,    show_default=True, type=int,   help="Number of traces to use.")
@click.option("--num-iter",    default=50,    show_default=True, type=int,   help="Maximum BP iterations.")
@click.option("--damping",     default=0.0,   show_default=True, type=float, help="Message damping factor (0=none, 0.5=recommended for t0-unknown).")
@click.option("--t0-known",    is_flag=True,  default=False,                 help="Use t0-known mode (default: t0-unknown).")
@click.option("--use-hint",    is_flag=True,  default=False,                 help="Use hint-bit constraint (default: no).")
@click.option("--traceset",    default=0,     type=int,                      help="Number of traceset")
def main(level, p_bit_error, num_traces, num_iter, damping, t0_known, use_hint, traceset):
    global ETA, TAU, DELTA, GAMMA_2
    if level == 2:
        ETA = 2
        TAU = 39
        DELTA = 44
        GAMMA_2 = (q-1)//(2*DELTA)
    elif level == 3:
        ETA = 4
        TAU = 49
        DELTA = 16
        GAMMA_2 = (q-1)//(2*DELTA)
    elif level == 5:
        ETA = 2
        TAU = 60
        DELTA = 16
        GAMMA_2 = (q-1)//(2*DELTA)
    else:
        print("Security level not defined")
        exit(-1)

    # Resolve --p-bit-error: the two sentinels select the per-bit measured error
    # rates (a list of RHO values, one per chi bit); anything else is a uniform
    # float rate applied to all bits (previous behavior).
    if not p_bit_error in ["USE_P_KC", "USE_P_DL"]:
        p_bit_error = float(p_bit_error)

    if t0_known:
        trace_file = f"traces/t0_known/traces_level{level}_t0_known_1000_{traceset}.pkl"
    else:
        trace_file = f"traces/t0_unknown/traces_level{level}_t0_unknown_1000_{traceset}.pkl"

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
            if t0_known == False:
                Azct1_low = pickle.load(f)
                h = pickle.load(f)
                list_traces.append((w, w1, w0, c, xD, Azct1_low, h))
            else:
                list_traces.append((w, w1, w0, c, xD, xD, xD))

    label = f"ML-DSA level {level} ({'t0-known' if t0_known else 't0-unknown'})"
    print(f"\n=== {label}  (eta={ETA}, tau={TAU}, traces={num_traces}, p_bit_error={p_bit_error}, damping={damping}, use_hint={use_hint}) ===")
    ok, rec, n_, elapsed = run_attack(
        p_bit_error, list_traces, s2, w0, t0,
        num_iterations=num_iter,t0_is_known=t0_known, damping=damping, use_hint=use_hint
    )
    print(f"  => correct={ok}/{n_} ({100*ok/n_:.1f}%)  recovered={rec}/{n_}  total {elapsed:.1f}s")


if __name__ == "__main__":
    main()

