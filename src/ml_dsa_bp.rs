use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

type Msg = HashMap<i32, f64>;
type FactorMsgs = Vec<Vec<(usize, Vec<(i32, f64)>)>>;

fn convadd(g: &Msg, s_msg: &Msg, c: i32) -> Msg {
    let mut out = Msg::new();
    for (&sv, &sp) in s_msg {
        for (&gv, &gp) in g {
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

// Cavity belief of variable j w.r.t. a specific factor:
//   cavity_log[v] = log_key_probs_j[v] - prev_msg[v]
// then softmax-normalized. When prev_msg is None (first iteration), uses
// log_key_probs_j directly (equivalent to full belief = uniform for zero prior).
fn cavity_belief(log_key_probs_j: &Msg, prev_msg: Option<&[(i32, f64)]>, n_vals: usize) -> Msg {
    let cavity_log: Msg = match prev_msg {
        Some(prev) => {
            // prev is tiny (2*eta+1 entries), linear scan avoids HashMap allocation
            log_key_probs_j
                .iter()
                .map(|(&v, &lp)| {
                    let pm = prev.iter().find(|&&(k, _)| k == v).map_or(0.0, |&(_, p)| p);
                    (v, lp - pm)
                })
                .collect()
        }
        None => log_key_probs_j.clone(),
    };
    let max_lp = cavity_log.values().cloned().fold(f64::NEG_INFINITY, f64::max);
    let raw: Msg = cavity_log.iter().map(|(&v, &l)| (v, (l - max_lp).exp())).collect();
    let sum: f64 = raw.values().sum();
    if sum > 0.0 {
        raw.into_iter().map(|(v, p)| (v, p / sum)).collect()
    } else {
        log_key_probs_j.keys().map(|&v| (v, 1.0 / n_vals as f64)).collect()
    }
}

// For each output i, builds left[k] and right[k] using cavity beliefs
// (log_key_probs minus this factor's previous message), then computes the
// factor→variable message for each connected s[j].
fn compute_contributions(
    trace: &Trace,
    log_key_probs: &[Msg],
    prev_msgs: &FactorMsgs,
    n: usize,
    eta: i32,
) -> FactorMsgs {
    let n_vals = (2 * eta + 1) as usize;
    (0..n)
        .into_par_iter()
        .map(|i| {
            let c_nz = get_nonzero_for_output(&trace.challenge, i);
            let t = c_nz.len();
            if t == 0 {
                return vec![];
            }
            let msg_x = &trace.x_priors[i];

            // prev_msgs[i][k] is in the same positional order as c_nz[k] — no lookup needed
            let prev_msgs_i: &[(usize, Vec<(i32, f64)>)] =
                if i < prev_msgs.len() { &prev_msgs[i] } else { &[] };

            // Cavity beliefs for each variable in c_nz, w.r.t. this factor (output i)
            let cav_beliefs: Vec<Msg> = c_nz
                .iter()
                .enumerate()
                .map(|(k, &(j, _))| {
                    let prev = prev_msgs_i.get(k).map(|(_, msg)| msg.as_slice());
                    cavity_belief(&log_key_probs[j], prev, n_vals)
                })
                .collect();

            // left[k] = dist of  sum_{l < k}  c_l * s[j_l]
            let mut left: Vec<Msg> = Vec::with_capacity(t);
            left.push({ let mut m = Msg::new(); m.insert(0, 1.0); m });
            for k in 0..t - 1 {
                let (_, ck) = c_nz[k];
                let next = convadd(&left[k], &cav_beliefs[k], ck);
                left.push(next);
            }

            // right[k] = dist of  sum_{l > k}  c_l * s[j_l]
            let mut right: Vec<Msg> = (0..t)
                .map(|_| { let mut m = Msg::new(); m.insert(0, 1.0); m })
                .collect();
            for k in (0..t - 1).rev() {
                let (_, ck) = c_nz[k + 1];
                right[k] = convadd(&right[k + 1], &cav_beliefs[k + 1], ck);
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
    prior: Vec<Msg>,
    log_key_probs: Vec<Msg>,
    prev_factor_msgs: Vec<FactorMsgs>,
}

#[pymethods]
impl MLDsaBP {
    #[new]
    pub fn new(n: usize, eta: i32) -> Self {
        let prior: Vec<Msg> = (0..n)
            .map(|_| (-eta..=eta).map(|s| (s, 0.0f64)).collect())
            .collect();
        let log_key_probs = prior.clone();
        MLDsaBP { n, eta, traces: Vec::new(), prior, log_key_probs, prev_factor_msgs: Vec::new() }
    }

    /// Set the prior distribution for each secret key coefficient.
    ///
    /// `prior` is a list of n dicts mapping value → probability (not log-prob).
    /// Resets log_key_probs to the new prior.
    pub fn set_prior(&mut self, prior: Vec<HashMap<i32, f64>>) -> PyResult<()> {
        if prior.len() != self.n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "prior must have length n",
            ));
        }
        for (j, prob_map) in prior.iter().enumerate() {
            for (&v, &p) in prob_map {
                if let Some(lp) = self.prior[j].get_mut(&v) {
                    *lp = if p > 0.0 { p.ln() } else { -1e300_f64 };
                }
            }
        }
        self.log_key_probs = self.prior.clone();
        Ok(())
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
        self.prev_factor_msgs.push(Vec::new());
        Ok(())
    }

    /// Run one BP iteration.
    ///
    /// Computes factor→variable messages using cavity beliefs (log_key_probs minus
    /// this factor's previous message) for each factor-variable pair, then
    /// replaces log_key_probs with the freshly accumulated values.
    pub fn run_iteration(&mut self) -> PyResult<()> {
        let mut new_log_key_probs: Vec<Msg> = self.prior.clone();
        let mut new_factor_msgs: Vec<FactorMsgs> = Vec::with_capacity(self.traces.len());

        let empty: FactorMsgs = Vec::new();
        for (t, trace) in self.traces.iter().enumerate() {
            let prev = self.prev_factor_msgs.get(t).unwrap_or(&empty);
            let contribs = compute_contributions(trace, &self.log_key_probs, prev, self.n, self.eta);
            for i_contribs in &contribs {
                for (j, deltas) in i_contribs {
                    for (sv, delta) in deltas {
                        *new_log_key_probs[*j].get_mut(sv).unwrap() += delta;
                    }
                }
            }
            new_factor_msgs.push(contribs);
        }

        self.log_key_probs = new_log_key_probs;
        self.prev_factor_msgs = new_factor_msgs;
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
        self.log_key_probs
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
    pub fn get_log_key_probs(&self) -> Vec<HashMap<i32, f64>> {
        self.log_key_probs.clone()
    }

    /// Reset log_key_probs to prior and discard all stored traces.
    pub fn reset(&mut self) {
        self.traces.clear();
        self.log_key_probs = self.prior.clone();
        self.prev_factor_msgs.clear();
    }

    pub fn trace_count(&self) -> usize {
        self.traces.len()
    }

    /// Clear stored factor→variable messages so the next run_iteration uses
    /// full beliefs instead of cavity beliefs (reproduces the old behaviour).
    pub fn clear_prev_messages(&mut self) {
        for msgs in &mut self.prev_factor_msgs {
            msgs.clear();
        }
    }
}
