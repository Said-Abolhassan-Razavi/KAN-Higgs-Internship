# KAN for Higgs Boson Classification
### University of Paris-Saclay — Internship Research

---

## Overview

This repository contains a complete, original implementation of **Kolmogorov-Arnold Networks (KAN)** for Higgs boson signal/background classification on the FAIR Universe HiggsML dataset.

The work builds upon and significantly extends prior work, with original contributions in model implementation, training methodology, systematic uncertainty robustness, and physics interpretability.

---

## Research Questions

1. Can KAN match or exceed XGBoost/MLP on tabular HEP data?
2. Does adaptive grid placement improve KAN performance?
3. Is KAN more robust under systematic energy scale uncertainties?
4. Can KAN's learned activation functions reveal physics structure?

---

## Key Contributions

| Contribution | Description |
|---|---|
| **PyTorch KAN from scratch** | Full B-spline implementation with Cox-de Boor recursion |
| **Adaptive grid update** | Redistributes knot positions to data quantiles during training |
| **Grid extension (coarse→fine)** | Progressive refinement from G=3 to G=10 |
| **L1 spline regularisation** | Encourages sparse, interpretable edge functions |
| **Mixed precision training** | AMP on CUDA for ~2x speedup |
| **MLP baseline** | Fair comparison: same parameter budget |
| **Bootstrap confidence intervals** | AUC and AMS reported as mean ± σ |
| **Systematic robustness** | Evaluates all models under energy scale shifts γ ∈ [0.8, 1.2] |
| **Adversarial decorrelation** | Pivoting method (Louppe et al., 2017) |
| **KAN interpretability** | Spline activation plots + feature importance |

---

## Dataset

**FAIR Universe — blackSwan_data** (systematic variation scenario)
- 2,000,000 events, 28 features
- Signal: H→ττ | Background: Z→ττ, ttbar, diboson
- Source: [FAIR Universe HiggsML Challenge](https://fair-universe.lbl.gov)

To download the dataset:
```bash
python download_data.py
```

---

## Environment

```
Python   3.12
PyTorch  2.11  (CUDA 12.8)
XGBoost  3.2
scikit-learn, pandas, pyarrow, seaborn, tqdm
```

Install all dependencies:
```bash
pip install torch xgboost scikit-learn pandas pyarrow seaborn tqdm
```

---

## How to Run

```bash
# 1. Download dataset
python download_data.py

# 2. Open notebook
jupyter lab KAN_Higgs_Internship.ipynb

# 3. Run all cells
```

For full results, set in the config cell:
```python
MAX_EVENTS  = None   # full 2M dataset
EPOCHS_MAIN = 300
```

> **Current status — smoke-test only:** The results below were produced on a subset of the data (`MAX_EVENTS = 100,000` out of 2,000,000 events, `EPOCHS_MAIN = 50`). Full results on the complete 2M-event dataset with 300 epochs will be added after the lab GPU run. To reproduce the full experiment set `MAX_EVENTS = None` and `EPOCHS_MAIN = 300` in the config cell.

---

## Results

All metrics are weighted AUC and AMS significance (mean ± σ over 200 bootstrap resamples on the held-out test set).

### Smoke-test: 100k events · 50 epochs

| Model | Params | AUC ± σ |
|---|---|---|
| XGBoost | — | 0.8812 ± 0.0007 |
| MLP | 14,529 | 0.8867 ± 0.0007 |
| KAN best config (G=5, k=4) | 34,689 | 0.8770 |

AMS significance and robustness metrics (Max AUC drop under energy scale shifts γ ∈ [0.8, 1.2]) for all 5 models will be reported after the full experiment on 2M events.

---

## Based On / Related Work

This work was developed independently during an internship at University of Paris-Saclay,
building upon and extending the following reference notebook:

> **Victor Borck** — *Notebook-Fair-Universe-KAN*
> GitHub: https://github.com/victorborck-ml/Notebook-Fair-Universe-KAN-Victor-Borck
>
> Victor's notebook provided the original exploration of KAN on the HiggsML dataset
> using TensorFlow. This repository is a complete rewrite in PyTorch with significant
> methodological improvements. See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for a detailed
> comparison.

---

## References

- Liu et al., *KAN: Kolmogorov-Arnold Networks*, arXiv:2404.19756 (2024)
- Liu et al., *KAN 2.0*, arXiv:2408.10205 (2024)
- Louppe et al., *Learning to Pivot with Adversarial Networks*, NeurIPS 2017
- Cowan et al., *Asymptotic formulae for likelihood-based tests*, arXiv:1007.1727
- FAIR Universe HiggsML Dataset, Zenodo:10.5281/zenodo.15131565