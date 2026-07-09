# Graph-Based Real-Time Anomaly Forecasting

## Key Properties

- **Unsupervised** — no anomaly labels required for training
- **Lightweight** — 4.9K parameters, 0.54 ms inference latency, 1,853 samples/sec throughput
- **Constant scaling** — parameter count and latency remain fixed as sensor count N and sequence length T increase
- **Fine-grained removal** — anomalies are removed per (timestep, sensor), preserving unaffected sensors at the same time step

---

## Hyperparameters

| Parameter | Value |
|---|---|
| GCN hidden dim (H) | 16 |
| GRU hidden dim (R) | 32 |
| Sequence length | 24 |
| Batch size | 64 |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Gradient clip norm | 1.0 |
| Early stopping patience | 20 epochs (val MSE) |
| Threshold multiplier k | 3 |

---

## Requirements

```
Python 3.8+
PyTorch 2.0+
PyTorch Geometric
numpy
pandas
scikit-learn
```

---

## Usage

```Jupyter Notebook
# 1. Put the dataset in a folder called data/
# 2. Prepare your adjacency matrix from SeReIn-M sensor groupings
# 3. Train GREAF on clean historical data - Each file is for each dataset. Run all the notebook cells
# 4. For other model comparison, run the 'comp-other-models.ipynb' file

```

---


