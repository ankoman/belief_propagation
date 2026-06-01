use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

type Msg = HashMap<i32, f64>;

fn convadd(g: &Msg, s_msg: &Msg, c: i32) -> Msg {
    let mut out = Msg::new();
    for (&gv, &gp) in g {
        for (&sv, &sp) in s_msg {
            *out.entry(gv + c * sv).or_insert(0.0) += gp * sp;
        }
    }
    out
}

// Coefficient of s[j] in x[i] = (s·c)[i] in Z[X]/(X^n+1)
fn challenge_weight(c: &[i32], i: usize, j: usize) -> i32 {
    let n = c.len();
    if j <= i {
        c[i - j]
    } else {
        -c[n + i - j]
    }
}

fn get_nonzero_for_output(c: &[i32], i: usize) -> Vec<(usize, i32)> {
    (0..c.len())
        .filter_map(|j| {
            let w = challenge_weight(c, i, j);
            if w != 0 { Some((j, w)) } else { None }
        })
        .collect()
}

/// Optimized belief propagation for the ML-DSA SASCA attack.
///
/// For each trace (challenge c, output priors x_priors), this computes
/// messages from Σ(c,i) factor nodes to each connected secret key variable s_j,
/// and accumulates log-probabilities. The MAP estimate s* = argmax Σ log-probs.
///
/// Uses the chain factor decomposition from Algorithm 2 in the paper, with
/// precomputed partial-sum distributions (valid since all challenge coefficients
/// are ±1 and s has a symmetric uniform prior).
#[pyclass]
pub struct MLDsaBP {
    n: usize,
    eta: i32,
    log_probs: Vec<Msg>,
}

#[pymethods]
impl MLDsaBP {
    #[new]
    pub fn new(n: usize, eta: i32) -> Self {
        let log_probs = (0..n)
            .map(|_| (-eta..=eta).map(|s| (s, 0.0f64)).collect())
            .collect();
        MLDsaBP { n, eta, log_probs }
    }

    /// Add one trace.
    ///
    /// challenge: list of n ints (challenge polynomial coefficients).
    /// x_priors: list of n dicts mapping x value → probability (measurement prior).
    pub fn add_trace(
        &mut self,
        challenge: Vec<i32>,
        x_priors: Vec<HashMap<i32, f64>>,
    ) -> PyResult<()> {
        if challenge.len() != self.n || x_priors.len() != self.n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "challenge and x_priors must have length n",
            ));
        }

        // Precompute partial-sum distributions with uniform symmetric s messages.
        // Since each challenge coeff is ±1 and s is symmetric uniform over [-η,η],
        // the partial sum distribution for k terms is the same regardless of ±1 signs.
        let tau = challenge.iter().filter(|&&v| v != 0).count();
        if tau == 0 {
            return Ok(());
        }
        let unif: Msg = {
            let p = 1.0 / (2 * self.eta + 1) as f64;
            (-self.eta..=self.eta).map(|s| (s, p)).collect()
        };
        let mut partial: Vec<Msg> = Vec::with_capacity(tau + 1);
        {
            let mut m = Msg::new();
            m.insert(0, 1.0);
            partial.push(m);
        }
        for _ in 0..tau {
            let nxt = convadd(partial.last().unwrap(), &unif, 1);
            partial.push(nxt);
        }

        // Parallel over output indices i (each is independent read; writes are
        // collected first, then applied serially to avoid data races).
        let contributions: Vec<Vec<(usize, Vec<(i32, f64)>)>> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let c_nz = get_nonzero_for_output(&challenge, i);
                let t = c_nz.len();
                if t == 0 {
                    return vec![];
                }
                let msg_x = &x_priors[i];
                c_nz.iter()
                    .enumerate()
                    .map(|(k, &(j, ck))| {
                        let g_left = &partial[k];
                        let h_right = &partial[t - k - 1];
                        let deltas: Vec<(i32, f64)> = (-self.eta..=self.eta)
                            .map(|sv| {
                                let sum: f64 = g_left
                                    .iter()
                                    .flat_map(|(&gl, &pl)| {
                                        h_right.iter().filter_map(move |(&gr, &pr)| {
                                            msg_x
                                                .get(&(gl + ck * sv + gr))
                                                .map(|&px| pl * pr * px)
                                        })
                                    })
                                    .sum();
                                let lp = if sum > 0.0 { sum.ln() } else { -1e300_f64 };
                                (sv, lp)
                            })
                            .collect();
                        (j, deltas)
                    })
                    .collect()
            })
            .collect();

        for i_contribs in contributions {
            for (j, deltas) in i_contribs {
                for (sv, delta) in deltas {
                    *self.log_probs[j].get_mut(&sv).unwrap() += delta;
                }
            }
        }
        Ok(())
    }

    /// Return the MAP estimate of the secret key polynomial (n coefficients).
    pub fn get_map_estimate(&self) -> Vec<i32> {
        self.log_probs
            .iter()
            .map(|lp| {
                lp.iter()
                    .max_by(|(_, a), (_, b)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(&v, _)| v)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Return accumulated log-probabilities for all secret key coefficients.
    pub fn get_log_probs(&self) -> Vec<HashMap<i32, f64>> {
        self.log_probs.clone()
    }

    /// Reset accumulated log-probabilities (keeps n and eta).
    pub fn reset(&mut self) {
        for lp in &mut self.log_probs {
            for v in lp.values_mut() {
                *v = 0.0;
            }
        }
    }
}
