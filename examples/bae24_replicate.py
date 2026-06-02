"""
Replication study: run the ML-DSA SASCA attack 10 times and report
average correct / recovered key counts.
"""

import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ml_dsa_attack import run_attack

# Attack configuration (ML-DSA-44)
N            = 256
ETA          = 2
TAU          = 39
NUM_TRACES   = 6
SNR          = 0.01
NUM_ITERS    = 20
NUM_RUNS     = 10

if __name__ == "__main__":
    for traces in range(100, 1001, 100):
        NUM_TRACES = traces
        print(f"ML-DSA-44  n={N} eta={ETA} tau={TAU} traces={NUM_TRACES} snr={SNR} iters={NUM_ITERS}")
        print(f"{'run':>4}  {'correct':>10}  {'recovered':>10}  {'time':>7}")
        print("-" * 40)

        total_ok  = 0
        total_rec = 0
        t_wall    = time.perf_counter()

        for run_idx in range(NUM_RUNS):
            ok, rec, n, elapsed = run_attack(
                N, ETA, TAU, NUM_TRACES, SNR,
                num_iterations=NUM_ITERS,
                seed=run_idx,
            )
            total_ok  += ok
            total_rec += rec
            print(f"{run_idx:>4}  {ok:>5}/{n} ({100*ok/n:5.1f}%)  {rec:>5}/{n} ({100*rec/n:5.1f}%)  {elapsed:6.1f}s")

        print("-" * 40)
        avg_ok  = total_ok  / NUM_RUNS
        avg_rec = total_rec / NUM_RUNS
        print(f" avg  {avg_ok:>5.1f}/{N} ({100*avg_ok/N:5.1f}%)  {avg_rec:>5.1f}/{N} ({100*avg_rec/N:5.1f}%)  "
            f"total {time.perf_counter()-t_wall:.1f}s")
