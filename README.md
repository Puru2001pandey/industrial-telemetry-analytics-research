# Industrial Telemetry Analytics Research — C‑MAPSS FD001

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22285377.svg)](https://doi.org/10.5281/zenodo.22285377)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Recall‑optimised failure detection in industrial telemetry.**  
This repository contains the corrected code and results for the paper:

> **Recall-Optimised Failure Detection in Industrial Telemetry:**  
> *A Rolling Degraded-State Counter Versus a 21-Sensor Random Forest on NASA C-MAPSS FD001*  
> — *Puru Pandey* (Preprint, revised September 2026, Version 3, [DOI: 10.5281/zenodo.22285377](https://doi.org/10.5281/zenodo.22285377))

---

## 🚨 Important: Corrected & Leak‑Free Pipeline

**This revision corrects the subtle look‑ahead bias** present in earlier exploratory versions of the paper.  
In the original draft, the degradation percentile threshold ($Q_{0.70}$) was computed over each engine's *full* run‑to‑failure trajectory, inadvertently using future (at‑risk) cycles to set the threshold.

**In this corrected implementation:**
- The threshold is estimated **strictly from healthy‑phase cycles** only (`RUL > 30 + 10` buffer).
- Two independent thresholding strategies are provided:
  1. **Global threshold** (pooled across all training engines' healthy cycles).
  2. **Per‑engine threshold** (each engine calibrated on its own healthy burn‑in data — more realistic for deployment).

---

## 🎯 Key Findings (Corrected Results)

| Model | At‑Risk Recall (seed 42) | Precision | Weighted F1 | Total Operational Cost* |
| :--- | :---: | :---: | :---: | ---: |
| **LR — Rolling Degraded Count (1 feature)** | **0.940** | 0.717 | 0.934 | **$3.00M** |
| **RF — 21 Raw Sensors (StandardScaler)** | 0.887 | 0.849 | 0.957 | $3.99M |
| *Baseline: Single Best Sensor (s11)* | 0.998 | 0.387 | 0.779 | $4.95M |
| *Baseline: Raw Composite Score* | 0.986 | 0.372 | 0.766 | $5.60M |
| *Baseline: Moving Average (w=10)* | 1.000 | 0.453 | 0.829 | $3.75M |

*\* Cost model: Missed failure (FN) = \$50,000 (unplanned downtime); False alarm (FP) = \$5,000 (unnecessary inspection). Illustrative assumed ratio, not a measured universal industrial figure.*  
*Test set: 20 held‑out engines, 3,852 cycles (620 at‑risk).*

### Key Insight
The rolling-count model is **neither the highest-recall nor the highest-precision model** in this benchmark. The three naive baselines achieve higher recall (0.986–1.000) but flood operators with false alarms (750–1,030 per test set, precision 0.372–0.453), while the Random Forest achieves higher precision (0.849) but misses more critical failures (70 misses vs. 37). The distinguishing property of the rolling degraded-state counter is that it avoids both high miss counts and high false alarms simultaneously, achieving the **lowest total operational cost (\$3.00M vs. \$3.99M for RF)** under the stated illustrative cost ratio.

---

## 📊 30‑Seed Cross‑Validation Stability (Per‑Engine Leak‑Free)

To ensure the result is not an artefact of a single train/test split, we repeated the experiment across 30 random 80/20 unit‑level splits:

| Metric | LR (Rolling Count) | RF (21 Sensors) |
| :--- | :---: | :---: |
| **Mean Recall** | **0.962** ± 0.024 | 0.898 ± 0.028 |
| **Range** | 0.910 – 1.000 | 0.840 – 0.958 |
| **Win Rate** | **28 / 30 splits** | — |
| **Paired t‑test** | **t = 9.36, p < 0.00000001** | — |

> **Conclusion:** The rolling-count model achieves the lowest cost-adjusted operating point under the illustrative \$50k/\$5k failure/inspection cost ratio. While three simpler baselines achieve higher raw recall, they suffer catastrophic precision collapse. Specifically against the 21-sensor Random Forest, the single rolling-count feature maintains a statistically significant recall advantage ($0.962 \pm 0.024$ vs. $0.898 \pm 0.028$, $p < 10^{-8}$) while avoiding excessive false alarms. Note that the cost model is an illustrative assumption rather than a universal industrial figure.

---

## 🧪 How to Reproduce

### 1. Download the data
*Note: `train_FD001.txt` is NOT redistributed in this repository due to original data licensing and must be obtained directly from NASA.*  
Download the archive from the [NASA C‑MAPSS dataset repository](https://data.nasa.gov/docs/legacy/CMAPSSData.zip), extract it, and place `train_FD001.txt` into the `data/` directory.

### 2. Run the corrected pipeline
```bash
python corrected_pipeline.py
```
This runs the global‑threshold leak‑free experiment (primary result, sensitivity grid, and 30‑seed stability).

### 3. Run the full confusion matrix & cost analysis
```bash
python run_prompts_5_6.py
```
This generates:
- Confusion matrices for LR, RF, and three baselines.
- Operational cost comparison under the \$50k/\$5k failure/inspection cost ratio.
- Per‑engine leak‑free calibration results (more realistic for live deployment).

### 4. View the raw results
All output logs and tables are saved in the `results/` folder:
- `results/confusion_matrices.txt`
- `results/per_engine_leak_free_results.txt`
- `summary.txt`

---

## 📁 Repository Structure
```
.
├── preprint_corrected_verified.pdf # Compiled paper PDF (Zenodo v3)
├── preprint_corrected_verified.tex # Full LaTeX paper source
├── corrected_pipeline.py          # Main leak‑free pipeline
├── run_prompts_5_6.py             # Confusion matrices + per‑engine results
├── data/                          # Place train_FD001.txt here
├── results/                       # Generated tables and logs
│   ├── confusion_matrices.txt     # Full confusion matrices & cost model
│   └── per_engine_leak_free_results.txt # Per-engine burn-in calibration
├── drafts/                        # Working drafts & write-ups (not part of pipeline)
│   ├── medium_blog_post_draft.md
│   ├── medium_paste.html
│   └── README.md
├── summary.txt                    # Comprehensive experiment summary
└── README.md                      # This file
```

---

## 📄 Citation

If you use this code or results, please cite the associated paper:

```bibtex
@misc{pandey2026recall,
  title={Recall-Optimised Failure Detection in Industrial Telemetry: A Rolling Degraded-State Counter Versus a 21-Sensor Random Forest on NASA C-MAPSS FD001},
  author={Pandey, Puru},
  year={2026},
  publisher={Zenodo},
  version={v3},
  doi={10.5281/zenodo.22285377},
  url={https://doi.org/10.5281/zenodo.22285377}
}
```

---

## 📬 Contact

- **Author:** Puru Pandey  
- **Email:** [purupandey2001@gmail.com](mailto:purupandey2001@gmail.com)  
- **GitHub:** [@Puru2001pandey](https://github.com/Puru2001pandey)

---

## License

MIT License — feel free to use and adapt with attribution.
