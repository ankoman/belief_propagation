A generic belief propagation implementation in Rust
=======

Initially created for the Chosen Ciphertext k-Trace Attacks on Masked CCA2 Secure Kyber paper (see https://eprint.iacr.org/2021/956.pdf) by Julius Hermelink, Silvan Streit, and Emanuele Strieder.

Below are modified by Junichi Sakamoto.

## Build

### Rust library

```bash
cargo build --release
```

### Python bindings

Python bindings are built with [maturin](https://github.com/PyO3/maturin).

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install maturin and build the extension module in-place
pip install maturin
maturin develop --release
```

After `maturin develop`, the `belief_propagation` module is importable from within the `.venv`.

### Run the ML-DSA attack example

```bash
source .venv/bin/activate
python examples/ml_dsa_attack.py
```
