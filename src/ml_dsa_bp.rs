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

// -----------------------------------------------------------------------
// Trace storage
// -----------------------------------------------------------------------

struct Trace {
    challenge: Vec<i32>,
    x_priors: Vec<Msg>,
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

fn beliefs_from_log_probs(log_probs: &[Msg], n_vals: usize) -> Vec<Msg> {
    log_probs
        .iter()
        .map(|lp| {
            let max_lp = lp.values().cloned().fold(f64::NEG_INFINITY, f64::max);
            let raw: Msg = lp.iter().map(|(&v, &l)| (v, (l - max_lp).exp())).collect();
            let sum: f64 = raw.values().sum();
            if sum > 0.0 {
                raw.into_iter().map(|(v, p)| (v, p / sum)).collect()
            } else {
                lp.keys().map(|&v| (v, 1.0 / n_vals as f64)).collect()
            }
        })
        .collect()
}

// Compute log-prob contributions from one trace.
//
// For each output i, builds left[k] and right[k] using `beliefs` (not uniform),
// then computes the factor→variable message for each connected s[j].
fn compute_contributions(
    trace: &Trace,
    beliefs: &[Msg],
    n: usize,
    eta: i32,
) -> Vec<Vec<(usize, Vec<(i32, f64)>)>> {
    (0..n)
        .into_par_iter()
        .map(|i| {
            let c_nz = get_nonzero_for_output(&trace.challenge, i);
            let t = c_nz.len();
            if t == 0 {
                return vec![];
            }
            let msg_x = &trace.x_priors[i];

            // left[k] = dist of  sum_{l < k}  c_l * s[j_l]
            let mut left: Vec<Msg> = Vec::with_capacity(t);
            left.push({ let mut m = Msg::new(); m.insert(0, 1.0); m });
            for k in 0..t - 1 {
                let (j, ck) = c_nz[k];
                let next = convadd(&left[k], &beliefs[j], ck);
                left.push(next);
            }

            // right[k] = dist of  sum_{l > k}  c_l * s[j_l]
            let mut right: Vec<Msg> = (0..t)
                .map(|_| { let mut m = Msg::new(); m.insert(0, 1.0); m })
                .collect();
            for k in (0..t - 1).rev() {
                let (j, ck) = c_nz[k + 1];
                right[k] = convadd(&right[k + 1], &beliefs[j], ck);
            }

            c_nz.iter()
                .enumerate()
                .map(|(k, &(j, ck))| {
                    let g_left = &left[k];
                    let h_right = &right[k];
                    let deltas: Vec<(i32, f64)> = (-eta..=eta)
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
        .collect()
}

// -----------------------------------------------------------------------
// Public struct
// -----------------------------------------------------------------------

#[pyclass]
pub struct MLDsaBP {
    n: usize,
    eta: i32,
    traces: Vec<Trace>,
    log_probs: Vec<Msg>,
}

#[pymethods]
impl MLDsaBP {
    #[new]
    pub fn new(n: usize, eta: i32) -> Self {
        let log_probs = (0..n)
            .map(|_| (-eta..=eta).map(|s| (s, 0.0f64)).collect())
            .collect();
        MLDsaBP { n, eta, traces: Vec::new(), log_probs }
    }

    /// Store a trace (challenge + measurement priors). Does not compute anything.
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
        self.traces.push(Trace { challenge, x_priors });
        Ok(())
    }

    /// Run one BP iteration.
    ///
    /// Computes factor→variable messages for every stored trace using the
    /// current variable beliefs (derived from log_probs), then replaces
    /// log_probs with the freshly accumulated values.
    ///
    /// On the first call log_probs is all-zero so beliefs are uniform,
    /// giving the same result as the old single-pass implementation.
    /// Subsequent calls use the updated beliefs, implementing true iterative BP.
    pub fn run_iteration(&mut self) -> PyResult<()> {
        let n_vals = (2 * self.eta + 1) as usize;
        let beliefs = beliefs_from_log_probs(&self.log_probs, n_vals);

        let mut new_log_probs: Vec<Msg> = (0..self.n)
            .map(|_| (-self.eta..=self.eta).map(|s| (s, 0.0f64)).collect())
            .collect();

        for trace in &self.traces {
            let contribs = compute_contributions(trace, &beliefs, self.n, self.eta);
            for i_contribs in contribs {
                for (j, deltas) in i_contribs {
                    for (sv, delta) in deltas {
                        *new_log_probs[j].get_mut(&sv).unwrap() += delta;
                    }
                }
            }
        }

        self.log_probs = new_log_probs;
        Ok(())
    }

    /// Convenience wrapper: run `iterations` BP iterations.
    pub fn propagate(&mut self, iterations: usize) -> PyResult<()> {
        for _ in 0..iterations {
            self.run_iteration()?;
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

    /// Reset log_probs and discard all stored traces.
    pub fn reset(&mut self) {
        self.traces.clear();
        for lp in &mut self.log_probs {
            for v in lp.values_mut() {
                *v = 0.0;
            }
        }
    }

    pub fn trace_count(&self) -> usize {
        self.traces.len()
    }
}
