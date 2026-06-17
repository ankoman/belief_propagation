use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::ml_dsa_bp::MLDsaBP;
use crate::variable_node::InputNeed;
use crate::{BPError, BPGraph, BPResult, NodeFunction, NodeIndex, Probability, VariableNode};

type T = i32;
type MsgT = HashMap<T, Probability>;
type Graph = BPGraph<T, MsgT>;

// -----------------------------------------------------------------------
// Factor node backed by a Python callable
// -----------------------------------------------------------------------

struct PyCallableFactorNode {
    py_func: Py<PyAny>,
    num_inputs: usize,
    connections: Option<Vec<NodeIndex>>,
}

// Py<PyAny> is Send + Sync in pyo3; all other fields are too.
// The derive would work as well, but being explicit is safer here.
unsafe impl Send for PyCallableFactorNode {}
unsafe impl Sync for PyCallableFactorNode {}

impl NodeFunction<T, MsgT> for PyCallableFactorNode {
    fn node_function(
        &mut self,
        inbox: Vec<(NodeIndex, MsgT)>,
    ) -> BPResult<Vec<(NodeIndex, MsgT)>> {
        let connections = self
            .connections
            .as_ref()
            .expect("PyCallableFactorNode not initialized");

        let n = inbox.len();

        // Re-order inbox entries to match connection order.
        let ordered: Vec<(NodeIndex, Vec<(T, Probability)>)> = connections
            .iter()
            .filter_map(|&conn| {
                inbox
                    .iter()
                    .find(|(idx, _)| *idx == conn)
                    .map(|(idx, msg)| {
                        let mut pairs: Vec<(T, Probability)> =
                            msg.iter().map(|(&v, &p)| (v, p)).collect();
                        pairs.sort_by_key(|(v, _)| *v);
                        (*idx, pairs)
                    })
            })
            .collect();

        if ordered.len() != n {
            return Err(BPError::new(
                "PyCallableFactorNode::node_function".to_owned(),
                format!(
                    "Expected {} inputs from connections, got {}",
                    n,
                    ordered.len()
                ),
            ));
        }

        let sizes: Vec<usize> = ordered.iter().map(|(_, v)| v.len()).collect();
        let total: usize = sizes.iter().product();

        // Pre-initialise every output message with 0.0 for all values in the
        // variable's support.  Without this, missing entries in sparse messages
        // are silently treated as probability 1 by mult_hashmaps, corrupting
        // get_result / get_marginal.
        let mut out_msgs: Vec<MsgT> = ordered
            .iter()
            .map(|(_, pairs)| pairs.iter().map(|(v, _)| (*v, 0.0_f64)).collect())
            .collect();

        // Iterate over every combination of values from the connected variables,
        // query the Python factor function, and accumulate sum-product messages.
        Python::with_gil(|py| -> BPResult<()> {
            let mut indices = vec![0usize; n];
            let mut done = total == 0;

            while !done {
                let current_vals: Vec<T> = (0..n).map(|i| ordered[i].1[indices[i]].0).collect();
                let current_probs: Vec<Probability> =
                    (0..n).map(|i| ordered[i].1[indices[i]].1).collect();

                let py_list = PyList::new_bound(py, &current_vals);
                let factor: f64 = self
                    .py_func
                    .bind(py)
                    .call1((py_list,))
                    .map_err(|e| {
                        BPError::new(
                            "PyCallableFactorNode::node_function".to_owned(),
                            format!("Python call failed: {e}"),
                        )
                    })?
                    .extract::<f64>()
                    .map_err(|e| {
                        BPError::new(
                            "PyCallableFactorNode::node_function".to_owned(),
                            format!("Could not extract float from Python return value: {e}"),
                        )
                    })?;

                if factor != 0.0 {
                    let prob_product: f64 = current_probs.iter().product();
                    for i in 0..n {
                        let prob_without_i = if current_probs[i].abs() > f64::EPSILON {
                            prob_product / current_probs[i]
                        } else {
                            current_probs
                                .iter()
                                .enumerate()
                                .filter(|(j, _)| *j != i)
                                .map(|(_, p)| p)
                                .product()
                        };
                        *out_msgs[i].entry(current_vals[i]).or_insert(0.0) +=
                            factor * prob_without_i;
                    }
                }

                // Increment mixed-radix counter.
                let mut carry = true;
                for i in (0..n).rev() {
                    if carry {
                        indices[i] += 1;
                        if indices[i] >= sizes[i] {
                            indices[i] = 0;
                        } else {
                            carry = false;
                        }
                    }
                }
                done = carry;
            }
            Ok(())
        })?;

        let result: Vec<(NodeIndex, MsgT)> = ordered
            .iter()
            .zip(out_msgs.into_iter())
            .map(|((idx, _), msg)| (*idx, msg))
            .collect();

        Ok(result)
    }

    fn is_factor(&self) -> bool {
        true
    }

    fn number_inputs(&self) -> Option<usize> {
        Some(self.num_inputs)
    }

    fn initialize(&mut self, connections: Vec<NodeIndex>) -> BPResult<()> {
        self.connections = Some(connections);
        Ok(())
    }

    fn is_ready(&self, recv_from: &Vec<(NodeIndex, MsgT)>, _step: usize) -> BPResult<bool> {
        Ok(recv_from.len() == self.num_inputs)
    }

    fn reset(&mut self) -> BPResult<()> {
        self.connections = None;
        Ok(())
    }

    fn get_prior(&self) -> Option<MsgT> {
        None
    }
}

// -----------------------------------------------------------------------
// Python-facing BPGraph wrapper
// -----------------------------------------------------------------------

#[pyclass(name = "BPGraph")]
pub struct PyBPGraph {
    inner: Graph,
    node_counter: usize,
}

#[pymethods]
impl PyBPGraph {
    #[new]
    pub fn new() -> Self {
        PyBPGraph {
            inner: BPGraph::new(),
            node_counter: 0,
        }
    }

    /// Add a variable node.  `prior` is an optional dict mapping i32 values to
    /// probabilities.  Returns the NodeIndex for use with add_edge / get_result.
    #[pyo3(signature = (prior=None))]
    pub fn add_variable_node(&mut self, prior: Option<HashMap<i32, f64>>) -> PyResult<usize> {
        let mut v: VariableNode<T, MsgT> = VariableNode::new();
        if let Some(p) = prior {
            v.set_prior(&p).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(e.to_string())
            })?;
        }
        let name = format!("v{}", self.node_counter);
        self.node_counter += 1;
        Ok(self.inner.add_node(name, Box::new(v)))
    }

    /// Add a factor node driven by a Python callable.
    ///
    /// `func` receives a list of i32 values (one per connected variable, in the
    /// order the edges were added) and must return a float (the factor
    /// probability / weight).  `num_inputs` must equal the number of add_edge
    /// calls that will connect to this node.
    pub fn add_factor_node(&mut self, func: Py<PyAny>, num_inputs: usize) -> usize {
        let factor = PyCallableFactorNode {
            py_func: func,
            num_inputs,
            connections: None,
        };
        let name = format!("f{}", self.node_counter);
        self.node_counter += 1;
        self.inner.add_node(name, Box::new(factor))
    }

    /// Connect a variable node and a factor node (order does not matter).
    pub fn add_edge(&mut self, node0: usize, node1: usize) -> PyResult<()> {
        self.inner.add_edge(node0, node1).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(e.to_string())
        })
    }

    /// Initialize all nodes (must be called before propagate).
    pub fn initialize(&mut self) -> PyResult<()> {
        self.inner.initialize().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Run `steps` rounds of belief propagation (single-threaded).
    pub fn propagate(&mut self, steps: usize) -> PyResult<()> {
        self.inner.propagate(steps).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Run one BP step (single-threaded).
    pub fn propagate_step(&mut self) -> PyResult<()> {
        self.inner.propagate_step().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Run `steps` rounds of belief propagation using `thread_count` threads.
    ///
    /// Note: Python factor nodes acquire the GIL for every combination of
    /// input values, so true parallelism only applies to pure-Rust factor
    /// nodes.  The GIL is released between scheduling operations.
    pub fn propagate_threaded(
        &mut self,
        py: Python<'_>,
        steps: usize,
        thread_count: u32,
    ) -> PyResult<()> {
        // Release the GIL so that worker threads can acquire it individually
        // when calling back into Python factor functions.
        py.allow_threads(|| {
            self.inner
                .propagate_threaded(steps, thread_count)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Return the marginal distribution at a variable node as a dict, or None
    /// if the result is not yet available.  Values are max-normalized (the
    /// largest entry equals 1.0); call normalize_result() to get a proper
    /// probability distribution.
    pub fn get_result(&self, node_index: usize) -> PyResult<Option<HashMap<i32, f64>>> {
        self.inner.get_result(node_index).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Like get_result but normalizes by the sum so values are true
    /// probabilities that sum to 1.
    pub fn get_marginal(&self, node_index: usize) -> PyResult<Option<HashMap<i32, f64>>> {
        let raw = self.inner.get_result(node_index).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })?;
        Ok(raw.map(|mut m| {
            let sum: f64 = m.values().sum();
            if sum > 0.0 {
                m.values_mut().for_each(|p| *p /= sum);
            }
            m
        }))
    }

    /// Return the most likely value at a variable node, or None if unavailable.
    pub fn get_map_estimate(&self, node_index: usize) -> PyResult<Option<i32>> {
        let raw = self.inner.get_result(node_index).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })?;
        Ok(raw.and_then(|m| {
            m.into_iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(v, _)| v)
        }))
    }

    /// Control whether outgoing messages are normalized after each step.
    pub fn set_normalize(&mut self, normalize: bool) {
        self.inner.set_normalize(normalize);
    }

    /// Reset the graph (clears all priors and messages; graph topology is kept).
    pub fn reset(&mut self) -> PyResult<()> {
        self.inner.reset().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    pub fn nodes_count(&self) -> usize {
        self.inner.nodes_count()
    }

    pub fn factor_nodes_count(&self) -> usize {
        self.inner.factor_nodes_count()
    }

    pub fn variable_nodes_count(&self) -> usize {
        self.inner.variable_nodes_count()
    }
}

// -----------------------------------------------------------------------
// Parallel gen_x_priors
// -----------------------------------------------------------------------

/// Compute x_priors for all n polynomial coefficients in parallel.
///
/// For each coefficient i, returns a dict mapping each v in [x_min, x_max] to
/// `(1 - p_bit_error) ** hamming_weight((v + xd_list[i]) XOR w0_obs_list[i] & mask)`.
///
/// This is the Rust+Rayon equivalent of the Python `gen_x_priors` inner loop in w0_attack.py.
#[pyfunction]
pub fn gen_x_priors_parallel(
    py: Python<'_>,
    w0_obs_list: Vec<i32>,
    xd_list: Vec<i32>,
    x_min: i32,
    x_max: i32,
    p_bit_error: f64,
    n_bits: u32,
) -> PyResult<Vec<HashMap<i32, f64>>> {
    if w0_obs_list.len() != xd_list.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "w0_obs_list and xd_list must have the same length",
        ));
    }
    let mask = (1i32 << n_bits) - 1;
    let base = 1.0_f64 - p_bit_error;

    let result = py.allow_threads(|| {
        (0..w0_obs_list.len())
            .into_par_iter()
            .map(|i| {
                let w0_obs = w0_obs_list[i];
                let xd_i = xd_list[i];
                (x_min..=x_max)
                    .map(|v| {
                        let hw = ((v.wrapping_add(xd_i)) ^ w0_obs) & mask;
                        let hw = hw.count_ones();
                        (v, base.powi(hw as i32))
                    })
                    .collect::<HashMap<i32, f64>>()
            })
            .collect::<Vec<_>>()
    });

    Ok(result)
}

// -----------------------------------------------------------------------
// Module entry point
// -----------------------------------------------------------------------

#[pymodule]
pub fn belief_propagation(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBPGraph>()?;
    m.add_class::<MLDsaBP>()?;
    m.add_function(wrap_pyfunction!(gen_x_priors_parallel, m)?)?;
    Ok(())
}
