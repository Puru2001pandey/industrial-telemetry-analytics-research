# Industrial Telemetry Analytics Research — C‑MAPSS FD001

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22304632.svg)](https://doi.org/10.5281/zenodo.22304632)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Recall‑optimised failure detection in industrial telemetry.**  
This repository contains the corrected code and results for the paper:

> **Recall-Optimised Failure Detection in Industrial Telemetry:**  
> *A Rolling Degraded-State Counter Versus a 21-Sensor Random Forest on NASA C-MAPSS FD001*  
> — *Puru Pandey* (Preprint, revised September 2026, Version 4, [DOI: 10.5281/zenodo.22304632](https://doi.org/10.5281/zenodo.22304632))

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
| **LR — Rolling Degraded Count (1 feature)** | **1.0000** | 0.5487 | 0.8812 | **$2.55M** |
| **RF — 21 Raw Sensors (StandardScaler)** | 0.8871 | 0.8488 | 0.9568 | $3.99M |
| *Baseline: Single Best Sensor (s11)* | 0.9984 | 0.3874 | 0.7790 | $4.95M |
| *Baseline: Raw Composite Score* | 1.0000 | 0.3695 | 0.7617 | $5.29M |
| *Baseline: Moving Average (w=10)* | 1.0000 | 0.4031 | 0.7926 | $4.59M |

*\* Cost model: Missed failure (FN) = \$50,000 (unplanned downtime); False alarm (FP) = \$5,000 (unnecessary inspection). Illustrative assumed ratio, not a measured universal industrial figure.*  
*Test set: 20 held‑out engines, 3,852 cycles (620 at‑risk).*

### Key Finding
- Four of five models reach at-risk recall 0.9984-1.0000, yet costs span $2.55M to $5.29M (2.07x). Recall does not distinguish them.
- The RF wins weighted F1 (0.9568) while missing 70 failures and costing 56% more than the cheapest model.
- Under a nested 60/20/20 protocol across 30 splits, the rolling-count model costs $2.11M +/- $0.39M vs $3.84M +/- $0.95M, 30/30 cost wins.
- Selecting (w, tau) on the test split understates cost by $790,167 (42%).
- The $50k/$5k ratio is an illustrative assumption, not a measured figure.

---

## 📊 30‑Seed Cross‑Validation Stability (Per‑Engine Leak‑Free)

To ensure the result is not an artefact of a single train/test split, we repeated the experiment across 30 random 80/20 unit‑level splits:

| Metric | LR (Rolling Count) | RF (21 Sensors) |
| :--- | :---: | :---: |
| **Mean Recall** | **1.0000** ± 0.0000 | 0.8977 ± 0.0284 |
| **Range** | 1.0000 – 1.0000 | 0.8403 – 0.9581 |
| **Win Rate** | **30 / 30 splits** | — |
| **Paired t‑test** | **t = 19.38, p < 10⁻¹⁵** | — |

> **Conclusion:** The rolling-count model achieves the lowest cost-adjusted operating point under the illustrative \$50k/\$5k failure/inspection cost ratio. While simpler baselines achieve comparable recall at severe precision penalties, the single rolling-count feature maintains a statistically significant recall advantage over the 21-sensor Random Forest ($1.0000 \pm 0.0000$ vs. $0.8977 \pm 0.0284$, $p < 10^{-15}$) while substantially reducing operational inspection costs. Note that the cost model is an illustrative assumption rather than a universal industrial figure.

---

## 🧪 How to Reproduce

### 1. Download the data
*Note: `train_FD001.txt` is NOT redistributed in this repository due to original data licensing and must be obtained directly from NASA.*  
Download the archive from the [NASA C‑MAPSS dataset repository](https://data.nasa.gov/docs/legacy/CMAPSSData.zip), extract it, and place `train_FD001.txt` into the `data/` directory.

### 2. Run the reproduction scripts
- **Primary result + sensitivity grid:**
  ```bash
  python corrected_pipeline.py
  ```
- **Confusion matrices + cost analysis (Table 1, Table 2):**
  ```bash
  python run_prompts_5_6.py
  ```
- **Cost grid, nested protocol, selection analyses (Table 4, Table 5, Sections 6.4–6.5):**
  ```bash
  python nested_cv.py
  ```

### 3. View the raw results
All output logs and tables are saved in the `results/` folder:
- `results/confusion_matrices.txt`
- `results/per_engine_leak_free_results.txt`
- `results/nested_cv_results.txt`
- `summary.txt`

---

## 📁 Repository Structure
```
.
├── preprint_corrected_verified.pdf # Compiled paper PDF (Zenodo v4)
├── preprint_corrected_verified.tex # Full LaTeX paper source
├── archive/                       # Superseded versions (retained for provenance)
│   ├── preprint_v3.pdf            # Zenodo v3 paper PDF (superseded)
│   ├── preprint_v3.tex            # Zenodo v3 LaTeX source
│   └── README.md
├── corrected_pipeline.py          # Primary result + sensitivity grid
├── run_prompts_5_6.py             # Confusion matrices + cost analysis
├── nested_cv.py                   # Cost grid, nested protocol, selection analyses
├── data/                          # Place train_FD001.txt here
├── results/                       # Generated tables and logs
│   ├── confusion_matrices.txt     # Full confusion matrices & cost model
│   ├── per_engine_leak_free_results.txt # Per-engine burn-in calibration
│   └── nested_cv_results.txt      # Nested CV, cost grid & optimism analyses
├── drafts/                        # Working drafts & write-ups (not part of pipeline)
│   ├── medium_blog_post_draft.md
│   ├── medium_paste.html
│   └── README.md
├── visuals/                       # Generated figures (regenerate from notebook)
├── summary.txt                    # Comprehensive experiment summary
└── README.md                      # This file
```

---

## 📄 Citation

If you use this code or results, please cite the associated paper:

```bibtex
@misc{pandey2026recall,
  title={When Recall Stops Discriminating: Cost-Adjusted Model Selection for Failure Detection in Industrial Telemetry on NASA C-MAPSS FD001},
  author={Pandey, Puru},
  year={2026},
  publisher={Zenodo},
  version={v4},
  doi={10.5281/zenodo.22304632},
  url={https://doi.org/10.5281/zenodo.22304632}
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
