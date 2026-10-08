import math
import random
import time
import sys, pickle
import numpy as np
import click
from statistics import NormalDist
from typing import Dict, List, Tuple
from ml_dsa_attack import count_recovered
from w0_attack import polyRing
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

def hard_to_soft(obs_chi, p_list):
    # Soft observation q_i = P(chi[i]=1 | L) of a hard-decision chi with per-bit
    # error rates p_list: q_i = 1 - p_i if the observed bit is 1, else p_i.
    return [1 - p_list[i] if (obs_chi >> i) & 1 else p_list[i] for i in range(RHO)]

def leak_soft_chi(chi, sigmas):
    # Soft-decision leakage of chi: each bit b leaks L = b + N(0, sigma_i^2). The
    # attacker knows the true leakage distribution (L | b ~ N(b, sigma_i^2)) and
    # the uniform bit prior, so the posterior is
    #   q_i = P(chi[i]=1 | L) = N(L; 1, sigma^2) / (N(L; 0, sigma^2) + N(L; 1, sigma^2)),
    # evaluated from the log-likelihoods to avoid underflow of the pdfs.
    q_bits = []
    for i in range(RHO):
        b = (chi >> i) & 1
        sigma = sigmas[i]

        L = b + random.gauss(0.0, sigma)
        ll0 = -(L - 0.0)**2 / (2 * sigma**2)   # log N(L; 0, sigma^2) + const
        ll1 = -(L - 1.0)**2 / (2 * sigma**2)   # log N(L; 1, sigma^2) + const
        m = max(ll0, ll1)
        q_bits.append(math.exp(ll1 - m) / (math.exp(ll0 - m) + math.exp(ll1 - m)))
    return q_bits


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

    # Phase 1: add traces with noisy observations
    for q1, w, w1, w0, c, xD, Azct1_low, h in list_traces:
        c.mod_pm()
        xD[attack_idx].mod_pm()
        x_priors = []
        q_bits_list = q1 #[leak_soft_chi(obs_SecDecomposeComp(w_i), sigmas) for w_i in w[attack_idx].coeff]
        bp.add_trace_from_leakage(
            list(c.coeff),
            list(w1[attack_idx].coeff),
            q_bits_list,
            list(xD[attack_idx].coeff),
            x_min, x_max,
            list(Azct1_low[attack_idx].coeff) if use_hint else [],
            list(h[attack_idx].coeff) if use_hint else [],
            U, V, TAU * ETA, DELTA,
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

    return ok, rec, n, elapsed
    

@click.command()
@click.option("--level",       "-l", default=2,     type=int,                      help="Security category in [2,3,5]")
@click.option("--num-traces",  "-n", default=50,    show_default=True, type=int,   help="Number of traces to use.")
@click.option("--num-iter",    "-i", default=50,    show_default=True, type=int,   help="Maximum BP iterations.")
@click.option("--damping",     "-d", default=0.0,   show_default=True, type=float, help="Message damping factor (0=none, 0.5=recommended for t0-unknown).")
@click.option("--t0-known",    is_flag=True,  default=False,                 help="Use t0-known mode (default: t0-unknown).")
@click.option("--use-hint",    is_flag=True,  default=False,                 help="Use hint-bit constraint (default: no).")
@click.option("--traceset",    "-s", default=0,     type=int,                      help="Number of traceset")
def main(level, num_traces, num_iter, damping, t0_known, use_hint, traceset):
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

    if t0_known:
        trace_file = f"traces/t0_known/traces_level{level}_t0_known_1000_{traceset}.pkl"
    else:
        trace_file = f"traces/t0_unknown/traces_level{level}_t0_unknown_1000_{traceset}.pkl"

    with open('p1_from_actual_tv0.pkl', 'rb') as f:
        p1 = pickle.load(f)

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
            # q1 make
            q1 = []
            for j in range(256):
                q_bits = []
                for k in range(RHO):
                    temp = 1.0
                    for share in range(4):
                        temp *= (1-2*p1[i*256+j][4*k+share])
                    q_bits.append((1-temp)/2)
                q1.append(q_bits)
            if t0_known == False:
                Azct1_low = pickle.load(f)
                h = pickle.load(f)
                list_traces.append((q1, w, w1, w0, c, xD, Azct1_low, h))
            else:
                list_traces.append((q1, w, w1, w0, c, xD, xD, xD))

    label = f"ML-DSA level {level} ({'t0-known' if t0_known else 't0-unknown'})"
    print(f"\n=== {label}  (eta={ETA}, tau={TAU}, traces={num_traces}, damping={damping}, use_hint={use_hint}) ===")
    ok, rec, n_, elapsed = run_attack(
        list_traces, s2, w0, t0,
        num_iterations=num_iter,t0_is_known=t0_known, damping=damping, use_hint=use_hint
    )
    print(f"  => correct={ok}/{n_} ({100*ok/n_:.1f}%)  recovered={rec}/{n_}  total {elapsed:.1f}s")


if __name__ == "__main__":
    main()

