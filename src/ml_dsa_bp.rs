use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

type Msg = HashMap<i32, f64>;
type FactorMsgs = Vec<Vec<(usize, Vec<(i32, f64)>)>>;

fn challenge_weight(c: &[i32], i: usize, j: usize) -> i32 {
    let n = c.len();
    if j <= i { c[i - j] } else { -c[n + i - j] }
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
// Dense distribution over a contiguous integer range
// -----------------------------------------------------------------------

#[derive(Clone)]
struct DenseMsg {
    offset: i32,    // key at data[0]
    data: Vec<f64>, // data[v - offset] = probability at key v
}

impl DenseMsg {
    fn delta(v: i32) -> Self {
        DenseMsg { offset: v, data: vec![1.0] }
    }

    fn from_sparse(map: &HashMap<i32, f64>) -> Self {
        if map.is_empty() {
            return DenseMsg { offset: 0, data: vec![] };
        }
        let lo = *map.keys().min().unwrap();
        let hi = *map.keys().max().unwrap();
        let mut data = vec![0.0f64; (hi - lo + 1) as usize];
        for (&k, &v) in map { data[(k - lo) as usize] = v; }
        DenseMsg { offset: lo, data }
    }
}

// -----------------------------------------------------------------------
// Convolution helpers  (ML-DSA: c ∈ {-1, 1}, |cav| = 2*eta+1)
// -----------------------------------------------------------------------

// out[gv + c·sv] += g[gv] · cav[sv + eta]   for gv in g, sv ∈ [-eta, eta].
// Output support: [g.offset - eta, g.offset + g.len - 1 + eta]  (valid for |c|=1).
//
// Inner kernel is a (2η+1)-tap SAXPY (5 or 9 elements for ML-DSA),
// written as a sequential slice-zip so LLVM auto-vectorises it.
fn convadd_dense(g: &DenseMsg, cav: &[f64], eta: i32, c: i32) -> DenseMsg {
    debug_assert!(c == 1 || c == -1, "ML-DSA challenge weights must be ±1");
    let n_cav = cav.len();                        // 2·eta + 1
    let out_lo = g.offset - eta;                  // same for c = ±1 since |c| = 1
    let mut out_data = vec![0.0f64; g.data.len() + 2 * eta as usize];

    if c == 1 {
        // out_idx = gv_idx + sv_idx  → sequential writes
        for (gv_idx, &gp) in g.data.iter().enumerate() {
            if gp == 0.0 { continue; }
            let slice = &mut out_data[gv_idx..gv_idx + n_cav];
            for (o, &cv) in slice.iter_mut().zip(cav) {
                *o += gp * cv;
            }
        }
    } else {
        // c = -1: out_idx = gv_idx + (2η - sv_idx).
        // Reversing cav keeps the write direction sequential → same SIMD benefit.
        for (gv_idx, &gp) in g.data.iter().enumerate() {
            if gp == 0.0 { continue; }
            let slice = &mut out_data[gv_idx..gv_idx + n_cav];
            for (o, &cv) in slice.iter_mut().zip(cav.iter().rev()) {
                *o += gp * cv;
            }
        }
    }

    DenseMsg { offset: out_lo, data: out_data }
}

// RX_{k-1}(u) = Σ_sv cav_k(sv)·RX_k(u + c·sv) = convadd with negated c.
fn convaddrev_dense(g: &DenseMsg, cav: &[f64], eta: i32, c: i32) -> DenseMsg {
    convadd_dense(g, cav, eta, -c)
}

// -----------------------------------------------------------------------
// Dot product with clipped index range
// -----------------------------------------------------------------------

// dot(a, b[b_start .. b_start + a.len()]), out-of-range entries → 0.
// When indices are fully in range (the common ML-DSA case) this reduces to
// a plain slice dot-product that LLVM vectorises.
fn dot_clipped(a: &[f64], b: &[f64], b_start: i32) -> f64 {
    let b_len = b.len() as i32;
    if b_start >= b_len || b_start + a.len() as i32 <= 0 {
        return 0.0;
    }
    let (a_sl, b_sl) = if b_start >= 0 {
        let bs = b_start as usize;
        let len = a.len().min(b.len() - bs);
        (&a[..len], &b[bs..bs + len])
    } else {
        let skip = (-b_start) as usize;
        if skip >= a.len() { return 0.0; }
        let len = (a.len() - skip).min(b.len());
        (&a[skip..skip + len], &b[..len])
    };
    a_sl.iter().zip(b_sl).map(|(&x, &y)| x * y).sum()
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

// Returns a Vec of length 2*eta+1 where index sv+eta holds the cavity
// probability at sv.  Dense representation avoids HashMap overhead for the
// 5- or 9-element cavity belief used by ML-DSA.
fn cavity_belief_dense(
    log_key_probs_j: &Msg,
    prev_msg: Option<&[(i32, f64)]>,
    eta: i32,
) -> Vec<f64> {
    let n_vals = (2 * eta + 1) as usize;
    let log_vals: Vec<f64> = (0..n_vals as i32)
        .map(|sv_idx| {
            let sv = sv_idx - eta;
            let lp = log_key_probs_j.get(&sv).copied().unwrap_or(f64::NEG_INFINITY);
            let pm = prev_msg
                .and_then(|p| p.iter().find(|&&(k, _)| k == sv).map(|&(_, v)| v))
                .unwrap_or(0.0);
            lp - pm
        })
        .collect();

    let max_lp = log_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_lp.is_infinite() {
        return vec![1.0 / n_vals as f64; n_vals];
    }

    let mut raw: Vec<f64> = log_vals.iter().map(|&l| (l - max_lp).exp()).collect();
    let sum: f64 = raw.iter().sum();
    if sum > 0.0 {
        for v in &mut raw { *v /= sum; }
        raw
    } else {
        vec![1.0 / n_vals as f64; n_vals]
    }
}

// For each output i:
//   Forward pass:  left[k] = dist of Σ_{l<k} c_l·s[j_l]   (DenseMsg, SAXPY kernel)
//   Backward pass: rx = RX_k where RX_k(u) = Σ_r right[k](r)·msg_x(u+r).
//                  Starts at msg_x; updated by convaddrev (convadd with -c).
//   Message for s[j_k]:  m_k(sv) = dot(left[k].data, rx.data[left[k].offset + ck·sv - rx.offset ..])
//                  — plain slice dot product; vectorised by LLVM.
fn compute_contributions(
    trace: &Trace,
    log_key_probs: &[Msg],
    prev_msgs: &FactorMsgs,
    n: usize,
    eta: i32,
) -> FactorMsgs {
    (0..n)
        .into_par_iter()
        .map(|i| {
            let c_nz = get_nonzero_for_output(&trace.challenge, i);
            let t = c_nz.len();
            if t == 0 { return vec![]; }

            let msg_x = DenseMsg::from_sparse(&trace.x_priors[i]);

            // prev_msgs[i][k] is in positional order matching c_nz[k]
            let prev_msgs_i: &[(usize, Vec<(i32, f64)>)] =
                if i < prev_msgs.len() { &prev_msgs[i] } else { &[] };

            // cav_beliefs[k][sv + eta] = cavity probability at sv ∈ [-eta, eta]
            let cav_beliefs: Vec<Vec<f64>> = c_nz
                .iter()
                .enumerate()
                .map(|(k, &(j, _))| {
                    let prev = prev_msgs_i.get(k).map(|(_, msg)| msg.as_slice());
                    cavity_belief_dense(&log_key_probs[j], prev, eta)
                })
                .collect();

            // Forward pass: left[k] = dist of Σ_{l<k} c_l·s[j_l]
            let mut left: Vec<DenseMsg> = Vec::with_capacity(t);
            left.push(DenseMsg::delta(0));
            for k in 0..t - 1 {
                let (_, ck) = c_nz[k];
                let next = convadd_dense(&left[k], &cav_beliefs[k], eta, ck);
                left.push(next);
            }

            // Backward pass: rx = RX_k, starts at msg_x.
            // m_k(sv) = dot(left[k], rx[left[k].offset + ck·sv - rx.offset ..])
            let mut rx = msg_x;
            let mut messages: Vec<(usize, Vec<(i32, f64)>)> =
                (0..t).map(|_| (0, vec![])).collect();

            for k in (0..t).rev() {
                let (j, ck) = c_nz[k];
                let b_base = left[k].offset - rx.offset;   // add ck*sv per sv
                let deltas: Vec<(i32, f64)> = (-eta..=eta)
                    .map(|sv| {
                        let sum = dot_clipped(&left[k].data, &rx.data, b_base + ck * sv);
                        let lp = if sum > 0.0 { sum.ln() } else { -1e300_f64 };
                        (sv, lp)
                    })
                    .collect();
                messages[k] = (j, deltas);
                if k > 0 {
                    rx = convaddrev_dense(&rx, &cav_beliefs[k], eta, ck);
                }
            }

            messages
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
