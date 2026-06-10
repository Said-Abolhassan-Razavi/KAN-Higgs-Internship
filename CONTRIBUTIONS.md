# Research Contributions
## KAN for Higgs Boson Classification — University of Paris-Saclay Internship

---

## Weaknesses of the Reference Notebook (Victor Borck)

> Reference: https://github.com/victorborck-ml/Notebook-Fair-Universe-KAN-Victor-Borck

Ordered from most critical to minor.

---

### CRITICAL

#### 1. Framework Not Runnable on Modern Hardware
Victor's notebook uses **TensorFlow/Keras**, which has no support for CUDA 12.8
on Windows (dropped after CUDA 11.x). The notebook cannot run on any modern
GPU setup without a complete rewrite.

> Impact: The entire codebase is non-functional on current hardware.

#### 2. Static Grid — Core KAN Feature Missing
The most important algorithmic feature of KAN (Liu et al., 2024) is the
**adaptive grid update**: knot positions should be redistributed to match the
actual data distribution during training, then spline weights re-fitted by
least squares to preserve the learned function.

Victor's implementation uses a fixed uniform grid from `[-3, 3]` for the
entire training run. This wastes spline capacity in empty regions and under-
resolves dense regions, producing a weaker model than KAN is capable of.

> Impact: Results do not reflect true KAN performance.

#### 3. Missing Weight Scaling Constant
The dataset weights must be multiplied by `K = 275.14233349053035` for correct
physics normalisation (matching the expected LHC luminosity). Victor applies
this constant but it is buried in a single line with no explanation.

Without it, the AMS significance and all class-balance calculations are
physically meaningless — the signal and background contribute with completely
wrong relative importance.

> Impact: All quantitative results (AMS, AUC) are incorrect if K is omitted.

#### 4. Missing Value Handling Absent
The HiggsML dataset uses **-999.0** as a sentinel for absent jet features
(when an event has 0 or 1 jets, features for the second jet are undefined).
Victor feeds these -999 values directly into the B-spline, which evaluates
them as physically real momentum values far outside the grid range, producing
garbage activations.

> Impact: Incorrect feature representation for ~40% of events (0-jet and 1-jet).

---

### MAJOR

#### 5. No Proper Baseline Comparison
Victor only compares KAN against XGBoost. There is no **MLP (standard dense
network)** baseline. Without it, it is impossible to determine whether any
performance difference comes from KAN specifically or simply from using a
neural network at all.

> Impact: Conclusions about KAN's advantage over "classical ML" are not
> scientifically justified.

#### 6. No Validation Set — No Early Stopping
Victor uses a single 75/25 train/test split with no held-out validation set.
The model trains for a fixed number of epochs with no mechanism to detect
overfitting. The test set is effectively used for model selection
(hyperparameter tuning), which causes data leakage.

> Impact: Reported test metrics are optimistically biased.

#### 7. No Statistical Uncertainty on Metrics
All reported AUC and AMS values are single-point estimates with no uncertainty
bounds. With ~1M events the statistical noise is small but the model variance
(across seeds, subsamples) is not quantified at all.

> Impact: Cannot determine whether differences between models are statistically
> significant.

#### 8. Fairness Analysis is Superficial
Despite the "FAIR Universe" framing, the robustness analysis only measures
variance across random data chunks. It does not test what the challenge
actually requires: stability under **nuisance parameter shifts** (e.g. tau
energy scale). There is no adversarial decorrelation, no pivoting, and no
proper systematic uncertainty analysis.

> Impact: Does not address the core scientific question of the challenge.

#### 9. No Spline Regularisation
The KAN paper proposes L1 regularisation on spline coefficients to encourage
sparsity, which is the prerequisite for interpretability (sparse networks can
be pruned and symbolically simplified). Victor's implementation applies no
regularisation, producing dense, uninterpretable networks.

> Impact: Loses KAN's primary advantage over MLP.

---

### MODERATE

#### 10. Grid Extension Not Implemented
The KAN paper's coarse-to-fine training strategy (start with few knots, extend
to more) is not implemented. Training directly on a fine grid tends to overfit
early before learning global structure.

#### 11. No Interpretability Analysis
KAN's main selling point — that each edge function can be visualised,
compared to known physics formulas, and simplified via symbolic regression —
is completely unused. The network is trained and then treated as a black box,
identical to an MLP in practice.

#### 12. XGBoost Baseline is Under-Tuned
XGBoost receives fixed hyperparameters (`max_depth=3`, `lr=0.05`,
`n_estimators=20000`) while KAN receives an extensive random search. This
makes the comparison unfair — XGBoost may be significantly stronger with
proper tuning.

#### 13. Hardcoded File Path
```python
filename = r"C:\Users\Utilisateur-IAE\Downloads\..."
```
The notebook only runs on Victor's personal machine. Anyone else must manually
hunt for and edit this path.

---

### MINOR

#### 14. Unfinished Sections
- Cell 50: `#ensembling` — empty, planned but never implemented
- Cell 41: exercises section — empty
- Cell 19: feature engineering — commented out ("for a second iteration")

#### 15. No Reproducibility Guarantee
Multiple random seeds are set in different places. TensorFlow global state is
not fully reset between runs even when `clear_session()` is called, making
exact reproduction of results unreliable.

#### 16. Significance Calculation Uses Fixed Bins
AMS is computed over 100 fixed bins in `[0, 1]`. For sparse high-score regions
this is too coarse. No adaptive binning is used.

---
---

## My Contributions (This Notebook)

### Technical Contributions

| Contribution | Description |
|---|---|
| **PyTorch rewrite** | Complete reimplementation in PyTorch — runs natively on GPU with CUDA 12.8. Zero framework compatibility issues. |
| **Adaptive grid update** | `KANLayer.update_grid()`: redistributes knot positions to data quantiles every N epochs, then refits spline weights by least squares. Directly implements the algorithm from Liu et al. (2024). |
| **Grid extension (coarse→fine)** | `KANLayer.extend_grid()`: starts training with G=3 knots, then extends to G=10 while preserving the learned function. Prevents early overfitting to grid artefacts. |
| **L1 spline regularisation** | `KANLayer.reg()`: penalises spline coefficient magnitude, encouraging sparse, interpretable edge functions. |
| **Mixed precision training** | Automatic Mixed Precision (AMP) via `torch.cuda.amp`. ~2x training speedup on Ampere/Blackwell GPUs (RTX 5060). |
| **Cosine LR schedule + gradient clipping** | Prevents training instability in spline weight updates. |
| **Early stopping on validation AUC** | Held-out validation set monitors generalisation; best checkpoint is restored automatically. |

### Scientific Contributions

| Contribution | Description |
|---|---|
| **Missing value imputation** | Replaces the sentinel value (−25.0 for blackSwan_data) with per-feature mean of valid values before any model sees the data. Fixes a silent bug present in Victor's notebook (which fed −999 raw into the B-spline). |
| **Weight handling** | blackSwan_data weights are already physics-normalised (∑w ≈ 105,718, matching 10 fb⁻¹). No K-scaling is needed here, unlike the original FAIR Universe parquet where Victor applies K=275.14. Class-balance re-weighting (155×) is applied only to the training set to equalise signal/background loss contributions. |
| **Bootstrap confidence intervals** | All AUC and AMS values reported as mean ± σ over 200 bootstrap resamples. Differences between models are statistically interpretable. |
| **Three-way data split** | Train / validation / test prevents data leakage from hyperparameter tuning. |
| **Systematic uncertainty robustness** | Evaluates all models under energy scale shifts γ ∈ [0.80, 1.20], directly measuring what the FAIR Universe challenge tests. |
| **Adversarial decorrelation** | Implements the pivoting method (Louppe et al., NeurIPS 2017): an adversary network tries to predict the nuisance from the classifier score; the classifier is trained to fool it. Makes the score approximately independent of the nuisance parameter. |
| **KAN interpretability** | Visualises the learned 1-D spline activation for every input feature. Shows knot positions before and after adaptive update. Computes feature importance from spline weight norms and compares against XGBoost gain. |
| **Hyperparameter sweep** | Systematic grid over `grid_size × spline_order` with AUC heatmap. Identifies the best KAN configuration for this dataset. |

### Engineering Contributions

| Contribution | Description |
|---|---|
| **Automatic data download** | `download_data.py` fetches the dataset from Zenodo with a progress bar. Anyone can reproduce the experiment with one command. |
| **`.gitignore`** | Ensures the 14 GB data file is never accidentally pushed to GitHub. |
| **Single config cell** | All hyperparameters defined in one place (Part 0-C). No hunting through cells to change a learning rate. |

---

## Research Questions Addressed

| Question | Addressed by Victor | Addressed Here |
|---|---|---|
| Can KAN match XGBoost? | Partially (no fair baseline) | Yes — with equal tuning and bootstrap CI |
| Does adaptive grid help? | No | Yes — grid update + extension experiment |
| Is KAN robust to systematics? | Proxy only | Yes — proper shift analysis + adversarial |
| Can KAN reveal physics? | No | Yes — spline visualisation + feature importance |
| Are results statistically significant? | No | Yes — bootstrap CI |

---

## References

- Liu et al., *KAN: Kolmogorov-Arnold Networks*, arXiv:2404.19756 (2024)
- Liu et al., *KAN 2.0*, arXiv:2408.10205 (2024)
- Louppe et al., *Learning to Pivot with Adversarial Networks*, NeurIPS 2017
- Cowan et al., *Asymptotic formulae for likelihood-based tests of new physics*, arXiv:1007.1727
- FAIR Universe HiggsML Challenge Dataset, Zenodo:10.5281/zenodo.15131565
