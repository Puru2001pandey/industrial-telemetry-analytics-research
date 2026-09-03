# Industrial Telemetry Analytics: Recall-Optimised Failure Detection on NASA C-MAPSS FD001

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22259839.svg)](https://doi.org/10.5281/zenodo.22259839)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-NASA%20CMAPSS-red)](https://www.nasa.gov/intelligent-systems-division/)

> **Author:** Puru Pandey — B.Tech AI/ML, Galgotias University  
> **Contact:** purupandey2001@gmail.com  
> **Preprint:** [Zenodo Record](https://zenodo.org/records/22259839) | **DOI:** [10.5281/zenodo.22259839](https://doi.org/10.5281/zenodo.22259839)

```bibtex
@article{pandey2026recall,
  title        = {Recall-Optimised Failure Detection in Industrial Telemetry: A Rolling Degraded-State Counter Versus a 21-Sensor Random Forest on NASA C-MAPSS FD001},
  author       = {Pandey, Puru},
  journal      = {Zenodo Preprint},
  year         = {2026},
  month        = {sep},
  doi          = {10.5281/zenodo.22259839},
  url          = {https://doi.org/10.5281/zenodo.22259839}
}
```

---

## Abstract

This study compares a logistic regression trained on a single rolling 10-cycle
degraded-state counter against a 100-tree Random Forest trained on all 21 raw sensor
channels from the NASA C-MAPSS FD001 benchmark (20,631 cycles, 100 engine units).
Both models are evaluated on a unit-level 80/20 split (no temporal leakage).
The primary metric is recall on the at-risk class — cycles within 30 of engine failure —
because in predictive maintenance, a missed failure costs far more than a false alarm.

---

## Key Findings

- A logistic regression on a **single rolling 10-cycle degraded-state counter** achieves **0.98 recall** on at-risk cycles (seed 42).
- A 100-tree Random Forest on **all 21 raw sensor channels** achieves **0.88 recall** on the same test set.
- The RF has higher weighted F1 (0.9576 vs. 0.9247), but this metric aggregates across classes and is misleading when missed failures cost orders of magnitude more than false alarms.
- Across 30 random seeds: LR mean recall **0.957 ± 0.018** vs. RF **0.916 ± 0.026** (paired t-test p < 0.0001; LR wins 29/30 splits).
- Sensitivity analysis across window lengths (5–20 cycles) and percentile thresholds (50–90%) confirms the LR recall advantage is not an artefact of a single lucky parameter setting.

---

## Repository Structure

```
industrial-telemetry-analytics-research/
│
├── README.md                    ← This file
├── arxiv_preprint.tex           ← Full LaTeX paper (published on Zenodo)
├── run_reproduction.ipynb       ← Standalone reproduction notebook (paper results)
├── analysis.ipynb               ← Full exploratory research notebook
│
├── data/
│   └── train_FD001.txt          ← NASA C-MAPSS FD001 (place here before running)
│
├── results/
│   ├── summary_metrics.csv      ← At-risk recall, precision, weighted F1 for both models
│   ├── confusion_matrix_lr.csv  ← LR confusion matrix (TN=3335, FP=336, FN=15, TP=605)
│   └── confusion_matrix_rf.csv  ← RF confusion matrix (TN=3561, FP=110, FN=74, TP=546)
│
└── requirements.txt
```

---

## How to Reproduce

> **Data is not committed to this repository.** Place `train_FD001.txt` in `data/`
> before running. The notebook raises `FileNotFoundError` if the file is absent.

```bash
# 1. Clone
git clone https://github.com/Puru2001pandey/industrial-telemetry-analytics-research.git
cd industrial-telemetry-analytics-research

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data
# train_FD001.txt is available from:
# https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/train_FD001.txt
# Place it in: data/train_FD001.txt

# 4. Reproduce paper results
jupyter notebook run_reproduction.ipynb
```

---

## Dataset

**NASA C-MAPSS FD001** — Commercial Modular Aero-Propulsion System Simulation

| Property | Value |
|---|---|
| Source | NASA Intelligent Systems Division |
| Rows | 20,631 operational cycles |
| Engine units | 100 (run-to-failure) |
| Sensor channels | 21 |
| Operating conditions | 1 (FD001 subset) |
| At-risk definition | Last 30 cycles before failure |
| Class balance | 85.6% healthy / 14.4% at-risk |

---

## Methodology Summary

```
Phase 1 — Data Ingestion
    Per-unit RUL calculation; binary label (at-risk = last 30 cycles)

Phase 2 — Sensor Preprocessing
    Retain 14 sensors with meaningful variance in FD001
    Per-unit min-max normalisation (Ramasso & Saxena 2014)
    Rising sensors: s02,s03,s04,s07,s08,s09,s11,s12,s13,s14,s15
    Falling sensors: s17,s20,s21

Phase 3 — Feature Engineering
    Degradation composite score (mean rising + inverted mean falling) / 2
    Binary degraded flag: score >= 70th-percentile of engine's distribution
    Health-window feature: count of degraded cycles in 10-cycle rolling window

Phase 4 — Classification
    Model A: L2-regularised logistic regression on health-window (1 feature)
    Model B: 100-tree Random Forest on all 21 normalised sensor channels
    Both: class_weight='balanced'; unit-level 80/20 split (seed 42)

Phase 5 — Evaluation
    Primary metric: at-risk recall
    Secondary: weighted F1, precision, confusion matrix
    Stability: 30-seed repeat with paired t-test
    Sensitivity: grid search over window length {5,10,15,20} and threshold {50-90%}
```

---

## Results

| Model | At-Risk Recall | At-Risk Precision | Weighted F1 |
|---|---|---|---|
| LR — health-window (1 feature) | **0.98** | 0.64 | 0.9247 |
| RF — 21 sensor channels | 0.88 | 0.83 | **0.9576** |

Confusion matrices (20 held-out engines, 4,291 test cycles):

| | Predicted Healthy | Predicted At-Risk |
|---|---|---|
| **LR — Actual Healthy** | 3,335 | 336 |
| **LR — Actual At-Risk** | **15** | 605 |
| **RF — Actual Healthy** | 3,561 | 110 |
| **RF — Actual At-Risk** | **74** | 546 |

---

## Limitations

- FD001 is the simplest C-MAPSS subset (single operating condition, single fault mode). Results may not transfer to FD002–FD004.
- The 70th-percentile threshold and 10-cycle window length are design choices, not derived from physical principles.
- The percentile is computed over full engine trajectories (post-mortem). Live deployment requires estimating the threshold from healthy-phase data only.
- No comparison against simpler baselines (single-sensor threshold, moving average).

---

## References

1. Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. *Proc. PHM*.
2. Ramasso, E. & Saxena, A. (2014). Performance benchmarking and analysis of prognostic methods for CMAPSS datasets. *IJPHM*, 5(2).
3. Li, X., Ding, Q., & Sun, J.-Q. (2018). Remaining useful life estimation using deep CNNs. *RESS*, 172, 1–11.
4. Mobley, R.K. (2002). *An Introduction to Predictive Maintenance*, 2nd ed.

---

## License

MIT License. See [LICENSE](LICENSE).

---

*Puru Pandey · purupandey2001@gmail.com · [LinkedIn](https://linkedin.com/in/puru-pandey-851271229) · [GitHub](https://github.com/Puru2001pandey)*
