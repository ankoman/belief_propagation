use pyo3::prelude::*;
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use std::cell::RefCell;
use std::collections::HashMap;

type Complex64 = Complex<f64>;

thread_local! {
    static FFT_PLANNER: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::new());
}

const FFT_THRESHOLD: usize = 65_536;

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
    offset: i32,
    data: Vec<f64>,
}

impl DenseMsg {
    fn delta(v: i32) -> Self {
        DenseMsg { offset: v, data: vec![1.0] }
    }

    // Only nonzero entries determine the dense range, so sparse priors
    // (e.g. a delta with a large surrounding zero-filled dict) stay compact.
    fn from_sparse(map: &HashMap<i32, f64>) -> Self {
        let lo_opt = map.iter().filter(|(_, &v)| v != 0.0).map(|(&k, _)| k).min();
        let hi_opt = map.iter().filter(|(_, &v)| v != 0.0).map(|(&k, _)| k).max();
        let (lo, hi) = match (lo_opt, hi_opt) {
            (Some(lo), Some(hi)) => (lo, hi),
            _ => return DenseMsg { offset: 0, data: vec![] },
        };
        let mut data = vec![0.0f64; (hi - lo + 1) as usize];
        for (&k, &v) in map {
            if v != 0.0 {
                data[(k - lo) as usize] = v;
            }
        }
        DenseMsg { offset: lo, data }
    }
}

// -----------------------------------------------------------------------
// Convolution helpers generalised to any secret range [s_min, s_max]
//
// out[gv + c·sv] += g[gv] · cav[sv - s_min]   for gv in g, sv ∈ [s_min, s_max].
//
// Output support:
//   c =  1: [g.offset + s_min,  g.offset + g.len - 1 + s_max]
//   c = -1: [g.offset - s_max,  g.offset + g.len - 1 - s_min]
// Both have length  g.data.len() + (s_max - s_min).
// -----------------------------------------------------------------------

fn convadd_dense(g: &DenseMsg, cav: &[f64], s_min: i32, s_max: i32, c: i32) -> DenseMsg {
    debug_assert!(c == 1 || c == -1, "ML-DSA challenge weights must be ±1");
    let n_cav = cav.len();
    let span  = (s_max - s_min) as usize;
    let out_lo = if c == 1 { g.offset + s_min } else { g.offset - s_max };
    let mut out_data = vec![0.0f64; g.data.len() + span];

    if c == 1 {
        for (gv_idx, &gp) in g.data.iter().enumerate() {
            if gp == 0.0 { continue; }
            let slice = &mut out_data[gv_idx..gv_idx + n_cav];
            for (o, &cv) in slice.iter_mut().zip(cav) {
                *o += gp * cv;
            }
        }
    } else {
        // c = -1: reversing cav keeps sequential write order → same SIMD benefit.
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

// -----------------------------------------------------------------------
// FFT-based convolution
// -----------------------------------------------------------------------

fn linear_conv_fft(a: &[f64], b: &[f64]) -> Vec<f64> {
    let out_len = a.len() + b.len() - 1;
    let n = out_len.next_power_of_two();
    let scale = 1.0 / n as f64;
    FFT_PLANNER.with(|p| {
        let mut p = p.borrow_mut();
        let fwd = p.plan_fft_forward(n);
        let inv = p.plan_fft_inverse(n);
        let mut ca: Vec<Complex64> = a.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        ca.resize(n, Complex64::new(0.0, 0.0));
        let mut cb: Vec<Complex64> = b.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        cb.resize(n, Complex64::new(0.0, 0.0));
        fwd.process(&mut ca);
        fwd.process(&mut cb);
        for i in 0..n { ca[i] *= cb[i]; }
        inv.process(&mut ca);
        ca[..out_len].iter().map(|c| c.re * scale).collect()
    })
}

fn convadd_fft(g: &DenseMsg, cav: &[f64], s_min: i32, s_max: i32, c: i32) -> DenseMsg {
    debug_assert!(c == 1 || c == -1);
    let out_lo = if c == 1 { g.offset + s_min } else { g.offset - s_max };
    let data = if c == 1 {
        linear_conv_fft(&g.data, cav)
    } else {
        let rev: Vec<f64> = cav.iter().copied().rev().collect();
        linear_conv_fft(&g.data, &rev)
    };
    DenseMsg { offset: out_lo, data }
}

fn convadd_adaptive(g: &DenseMsg, cav: &[f64], s_min: i32, s_max: i32, c: i32) -> DenseMsg {
    if g.data.len() * cav.len() > FFT_THRESHOLD {
        convadd_fft(g, cav, s_min, s_max, c)
    } else {
        convadd_dense(g, cav, s_min, s_max, c)
    }
}

fn convaddrev_adaptive(g: &DenseMsg, cav: &[f64], s_min: i32, s_max: i32, c: i32) -> DenseMsg {
    convadd_adaptive(g, cav, s_min, s_max, -c)
}

// -----------------------------------------------------------------------
// Dot product with clipped index range
// -----------------------------------------------------------------------

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

fn compute_messages(
    left: &DenseMsg,
    rx: &DenseMsg,
    s_min: i32,
    s_max: i32,
    ck: i32,
) -> Vec<(i32, f64)> {
    let b_base = left.offset - rx.offset;
    let n_cav = (s_max - s_min + 1) as usize;

    if left.data.len() * n_cav > FFT_THRESHOLD {
        // FFT cross-correlation: corr(left, rx)[d] = Σ_i left[i] * rx[i + d]
        let n_out = left.data.len() + rx.data.len();
        let n = n_out.next_power_of_two();
        let scale = 1.0 / n as f64;
        FFT_PLANNER.with(|p| {
            let mut p = p.borrow_mut();
            let fwd = p.plan_fft_forward(n);
            let inv = p.plan_fft_inverse(n);
            let mut ca: Vec<Complex64> =
                left.data.iter().map(|&x| Complex64::new(x, 0.0)).collect();
            ca.resize(n, Complex64::new(0.0, 0.0));
            let mut cb: Vec<Complex64> =
                rx.data.iter().map(|&x| Complex64::new(x, 0.0)).collect();
            cb.resize(n, Complex64::new(0.0, 0.0));
            fwd.process(&mut ca);
            fwd.process(&mut cb);
            for i in 0..n { ca[i] = ca[i].conj() * cb[i]; }
            inv.process(&mut ca);
            (s_min..=s_max)
                .map(|sv| {
                    let d = b_base + ck * sv;
                    let la = left.data.len() as i32;
                    let lb = rx.data.len() as i32;
                    let sum = if d >= -(la - 1) && d < lb {
                        let d_wrap = ((d % n as i32) + n as i32) as usize % n;
                        ca[d_wrap].re * scale
                    } else {
                        0.0
                    };
                    let lp = if sum > 0.0 { sum.ln() } else { -1e300_f64 };
                    (sv, lp)
                })
                .collect()
        })
    } else {
        (s_min..=s_max)
            .map(|sv| {
                let sum = dot_clipped(&left.data, &rx.data, b_base + ck * sv);
                let lp = if sum > 0.0 { sum.ln() } else { -1e300_f64 };
                (sv, lp)
            })
            .collect()
    }
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

fn cavity_belief_dense(
    log_key_probs_j: &Msg,
    prev_msg: Option<&[(i32, f64)]>,
    s_min: i32,
    s_max: i32,
) -> Vec<f64> {
    let n_vals = (s_max - s_min + 1) as usize;
    // Build a map for O(1) prev-message lookup (important when range is wide).
    let prev_map: HashMap<i32, f64> = prev_msg
        .map(|p| p.iter().copied().collect())
        .unwrap_or_default();

    let log_vals: Vec<f64> = (s_min..=s_max)
        .map(|sv| {
            let lp = log_key_probs_j.get(&sv).copied().unwrap_or(f64::NEG_INFINITY);
            let pm = prev_map.get(&sv).copied().unwrap_or(0.0);
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

fn compute_contributions(
    trace: &Trace,
    log_key_probs: &[Msg],
    prev_msgs: &FactorMsgs,
    n: usize,
    s_min: i32,
    s_max: i32,
) -> FactorMsgs {
    (0..n)
        .into_par_iter()
        .map(|i| {
            let c_nz = get_nonzero_for_output(&trace.challenge, i);
            let t = c_nz.len();
            if t == 0 { return vec![]; }

            let msg_x = DenseMsg::from_sparse(&trace.x_priors[i]);

            let prev_msgs_i: &[(usize, Vec<(i32, f64)>)] =
                if i < prev_msgs.len() { &prev_msgs[i] } else { &[] };

            let cav_beliefs: Vec<Vec<f64>> = c_nz
                .iter()
                .enumerate()
                .map(|(k, &(j, _))| {
                    let prev = prev_msgs_i.get(k).map(|(_, msg)| msg.as_slice());
                    cavity_belief_dense(&log_key_probs[j], prev, s_min, s_max)
                })
                .collect();

            // Forward pass: left[k] = dist of Σ_{l<k} c_l·s[j_l]
            let mut left: Vec<DenseMsg> = Vec::with_capacity(t);
            left.push(DenseMsg::delta(0));
            for k in 0..t - 1 {
                let (_, ck) = c_nz[k];
                let next = convadd_adaptive(&left[k], &cav_beliefs[k], s_min, s_max, ck);
                left.push(next);
            }

            // Backward pass: rx = RX_k, starts at msg_x.
            let mut rx = msg_x;
            let mut messages: Vec<(usize, Vec<(i32, f64)>)> =
                (0..t).map(|_| (0, vec![])).collect();

            for k in (0..t).rev() {
                let (j, ck) = c_nz[k];
                let deltas = compute_messages(&left[k], &rx, s_min, s_max, ck);
                messages[k] = (j, deltas);
                if k > 0 {
                    rx = convaddrev_adaptive(&rx, &cav_beliefs[k], s_min, s_max, ck);
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
    s_min: i32,
    s_max: i32,
    traces: Vec<Trace>,
    prior: Vec<Msg>,
    log_key_probs: Vec<Msg>,
    prev_factor_msgs: Vec<FactorMsgs>,
    /// Fraction of the old message kept when mixing new and previous factor messages.
    /// 0.0 = no damping (pure new message), 0.5 = equal mix, 0.9 = very conservative.
    damping: f64,
}

#[pymethods]
impl MLDsaBP {
    #[new]
    pub fn new(n: usize, eta: i32) -> Self {
        let prior: Vec<Msg> = (0..n)
            .map(|_| (-eta..=eta).map(|s| (s, 0.0f64)).collect())
            .collect();
        let log_key_probs = prior.clone();
        MLDsaBP {
            n, eta, s_min: -eta, s_max: eta,
            traces: Vec::new(), prior, log_key_probs, prev_factor_msgs: Vec::new(),
            damping: 0.0,
        }
    }

    /// Set the message damping factor (0.0 = no damping, 1.0 = freeze).
    /// Damped message = log((1-d)*exp(new) + d*exp(old)).
    /// Recommended: 0.5 for the t0-unknown attack with a wide secret range.
    pub fn set_damping(&mut self, damping: f64) {
        self.damping = damping.clamp(0.0, 1.0);
    }

    /// Set the prior distribution for each secret key coefficient.
    ///
    /// `prior` is a list of n dicts mapping value → probability (not log-prob).
    /// The secret range [s_min, s_max] is inferred from the keys present in the
    /// prior, so this works for any range — not just [-eta, eta].
    pub fn set_prior(&mut self, prior: Vec<HashMap<i32, f64>>) -> PyResult<()> {
        if prior.len() != self.n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "prior must have length n",
            ));
        }

        // Infer the secret range from every key in the provided prior.
        let mut new_s_min = i32::MAX;
        let mut new_s_max = i32::MIN;
        for prob_map in &prior {
            for &v in prob_map.keys() {
                if v < new_s_min { new_s_min = v; }
                if v > new_s_max { new_s_max = v; }
            }
        }
        if new_s_min <= new_s_max {
            self.s_min = new_s_min;
            self.s_max = new_s_max;
        }

        // Completely replace each variable's prior map (do not filter by old keys).
        for (j, prob_map) in prior.iter().enumerate() {
            self.prior[j] = prob_map.iter()
                .map(|(&v, &p)| (v, if p > 0.0 { p.ln() } else { -1e300_f64 }))
                .collect();
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
    pub fn run_iteration(&mut self) -> PyResult<()> {
        let log_key_probs = &self.log_key_probs;
        let prev_factor_msgs = &self.prev_factor_msgs;
        let empty: FactorMsgs = Vec::new();
        let n = self.n;
        let s_min = self.s_min;
        let s_max = self.s_max;

        // Compute all trace contributions in parallel (traces are independent).
        let mut new_factor_msgs: Vec<FactorMsgs> = self.traces
            .par_iter()
            .enumerate()
            .map(|(t, trace)| {
                let prev = prev_factor_msgs.get(t).unwrap_or(&empty);
                compute_contributions(trace, log_key_probs, prev, n, s_min, s_max)
            })
            .collect();

        // Apply message damping: mix new and previous messages in probability space.
        // damped = log((1-d)*exp(new) + d*exp(prev))
        if self.damping > 0.0 {
            let log_new_w  = (1.0 - self.damping).ln();
            let log_prev_w = self.damping.ln();
            for (t, trace_msgs) in new_factor_msgs.iter_mut().enumerate() {
                let prev = match self.prev_factor_msgs.get(t) {
                    Some(p) if !p.is_empty() => p,
                    _ => continue,
                };
                for (i, output_msgs) in trace_msgs.iter_mut().enumerate() {
                    let prev_i = match prev.get(i) {
                        Some(p) => p,
                        None => continue,
                    };
                    for (k, (_, new_deltas)) in output_msgs.iter_mut().enumerate() {
                        let prev_deltas = match prev_i.get(k) {
                            Some((_, d)) => d,
                            None => continue,
                        };
                        for (idx, (_, new_lp)) in new_deltas.iter_mut().enumerate() {
                            if let Some((_, prev_lp)) = prev_deltas.get(idx) {
                                let a = log_new_w  + *new_lp;
                                let b = log_prev_w + *prev_lp;
                                let m = a.max(b);
                                *new_lp = m + ((a - m).exp() + (b - m).exp()).ln();
                            }
                        }
                    }
                }
            }
        }

        let mut new_log_key_probs: Vec<Msg> = self.prior.clone();
        for contribs in &new_factor_msgs {
            for i_contribs in contribs {
                for (j, deltas) in i_contribs {
                    for (sv, delta) in deltas {
                        if let Some(lp) = new_log_key_probs[*j].get_mut(sv) {
                            *lp += delta;
                        }
                    }
                }
            }
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
    /// full beliefs instead of cavity beliefs.
    pub fn clear_prev_messages(&mut self) {
        for msgs in &mut self.prev_factor_msgs {
            msgs.clear();
        }
    }
}
