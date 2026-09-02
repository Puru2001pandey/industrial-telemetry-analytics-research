# From Raw Telemetry to Operational Intelligence
## A Data Analytics Study on Industrial Machine Failure Patterns
**By Puru Pandey** | Data Analyst & AI/ML Researcher | [GitHub](https://github.com/Puru2001pandey) | [LinkedIn](https://linkedin.com/in/puru-pandey-851271229)

---

*Reading time: ~8 minutes*

---

When I was assigned a forensic data analytics project during my Deloitte Australia technology simulation, I was handed something deceptively simple: a folder of raw JSON files, each packed with sensor readings from industrial machines spread across four factories on different continents.

No documentation. No schema definition. Just gigabytes of telemetry — and the question: *"Which factory has a problem?"*

What followed was one of the most practical data research exercises I've done. In this post, I'll walk through the methodology, the key analytical decisions, and how I adapted these techniques using a publicly available NASA dataset so you can replicate the approach yourself.

---

## 🏭 The Problem: Industrial Telemetry at Scale

Modern factories generate continuous streams of sensor data — temperature, vibration, pressure, rotation speed — from dozens of machines, every second, around the clock. This data is invaluable for predictive maintenance, but only if you can make sense of it.

Our dataset contained:
- **60+ MB** of raw JSON telemetry
- **4 factories** across different geographic regions
- **9 machine types**, each with different sensor profiles
- **30 days** of operational records

The challenge: identify which factory had the highest machine failure rate, and *why*.

---

## 📊 Step 1: Schema Normalization — Making Sense of the Mess

The first obstacle was the data itself. Raw JSON telemetry is notoriously inconsistent — different machines report at different intervals, sensors have different naming conventions, and timestamps can be misaligned across time zones.

My preprocessing pipeline involved:

1. **Schema inference** — parsing nested JSON to extract machine ID, timestamp, and sensor readings into a flat tabular structure
2. **Type normalization** — ensuring numeric sensor values weren't stored as strings (a common JSON export issue)
3. **Temporal alignment** — converting all timestamps to UTC and creating a unified operational cycle index

```python
# Schema normalization — convert raw telemetry JSON to structured DataFrame
import json, pandas as pd

def normalize_telemetry(json_path: str) -> pd.DataFrame:
    with open(json_path) as f:
        raw = json.load(f)
    records = []
    for machine_id, cycles in raw.items():
        for cycle in cycles:
            records.append({
                'machine_id': machine_id,
                'timestamp': pd.to_datetime(cycle['ts']),
                **{k: float(v) for k, v in cycle['sensors'].items()}
            })
    return pd.DataFrame(records).sort_values(['machine_id', 'timestamp'])
```

After normalization, we had a clean analytical dataset: **one row per machine per time cycle**, with typed sensor columns.

---

## 🔬 Step 2: Exploratory Data Analysis — Finding the Signal

With clean data in hand, I moved into exploratory analysis. The key question: *which factory, and which machine type, showed the earliest signs of failure?*

I computed **mean operational lifetime** — the average number of cycles a machine ran before hitting a failure threshold — grouped by factory. The results were clear: one factory consistently produced machines with ~35% shorter operational lifetimes compared to the others.

```python
# Compute average machine lifetime by factory
unit_lifetimes = df.groupby(['factory', 'machine_id'])['cycle'].max().reset_index()
factory_avg = unit_lifetimes.groupby('factory')['cycle'].mean().sort_values()
print(factory_avg)
# Output:
# Factory_C    187.3   ← Highest failure rate (shortest lifetime)
# Factory_A    241.8
# Factory_B    256.2
# Factory_D    278.5
```

But the *why* was more interesting than the *which*. Digging into sensor correlation data revealed that machines in the high-failure factory were operating under more extreme temperature and vibration conditions — a systemic maintenance issue, not a hardware defect.

---

## ⚙️ Step 3: The Key Innovation — Custom Downtime Interval Encoding

The most impactful piece of this analysis was a custom feature I engineered: **discretized 10-minute downtime intervals**.

Here's the problem it solves: raw sensor readings are continuous and noisy. You can't look at a sensor value and immediately know "this machine is about to fail." But if you classify each reading into a health state (HEALTHY → DEGRADED → CRITICAL) and then count how many *10-minute intervals* a machine spent in a degraded state, you get a clean, interpretable signal.

```python
def engineer_downtime_intervals(df, health_col='health_index',
                                 unhealthy_threshold=0.35,
                                 interval_minutes=10):
    df = df.copy()
    # Step 1: Flag unhealthy cycles
    df['is_unhealthy'] = (df[health_col] < unhealthy_threshold).astype(int)
    # Step 2: Discretize into 10-minute buckets
    df['downtime_interval'] = (df['cycle'] // interval_minutes).astype(int)
    # Step 3: Cumulative downtime per machine
    df['cumulative_downtime'] = (
        df.groupby('machine_id')['is_unhealthy']
          .cumsum()
          .divide(interval_minutes)
          .apply(np.floor)
    )
    return df
```

This simple feature transformed a messy continuous signal into a clean metric: *"Machine X spent 14 unhealthy 10-minute intervals in the last week"*. That's something a factory manager can act on immediately.

---

### Model Comparison: Does One Feature Beat 21 Sensors?

To validate this approach on public data, I ran a controlled comparison on the **NASA C-MAPSS FD001** dataset (20,631 cycles, 100 engine units, 21 sensor channels). I trained two models on an 80/20 unit-level split (no temporal leakage):

- **Model A:** Logistic regression on the rolling 10-cycle degraded-state counter (1 feature)
- **Model B:** Random Forest on all 21 raw normalised sensor channels

```python
# Unit-level split — no data leakage across engines
np.random.seed(42)
units = df['unit_id'].unique(); np.random.shuffle(units)
train_u, test_u = set(units[:80]), set(units[80:])

# Model A: LR on health-window feature
lr = LogisticRegression(class_weight='balanced', random_state=42)
lr.fit(X_hw_train, y_train)
# At-risk recall: 0.98 | Weighted F1: 0.9247

# Model B: RF on all 21 sensors
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_rf_train, y_train)
# At-risk recall: 0.88 | Weighted F1: 0.9576
```

| Model | At-Risk Recall | Weighted F1 |
|---|---|---|
| LR — 1 health-window feature | **0.98** | 0.9247 |
| RF — 21 raw sensors | 0.88 | 0.9576 |

The Random Forest achieves higher overall weighted F1. But here is why that is the wrong metric to optimise: in predictive maintenance, **a missed failure causes unplanned downtime and potential safety incidents**, while a false alarm is recoverable (an unnecessary inspection). The primary metric is therefore recall on the at-risk class — and the single interpretable feature wins there by 10 percentage points.

This is the core lesson: metric selection is a domain decision, not a modelling decision. You choose what to optimise *before* you train, based on the cost asymmetry of your errors.

---

## 🔍 Step 4: Forensic Classification — The Compensation Strand

The second part of the project was entirely different in domain but used the same analytical mindset: classify gender pay equity scores across all job roles in a corporate compensation dataset.

The approach was straightforward:
1. Compute an **expected compensation** for each role, controlling for seniority and performance
2. Calculate each employee's **pay equity ratio** (actual / expected)
3. Apply classification to bucket employees into equity categories

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)
# Weighted F1 Score: 0.873
```

The classification achieved an F1 score of 0.873, reliably identifying employees categorized as underpaid relative to their role expectations. The findings revealed statistically meaningful disparities concentrated in specific role hierarchies — exactly the kind of evidence-based output that informs policy change.

---

## 💡 Key Learnings

**1. Metric selection is a domain decision, not a modelling decision.** A 21-sensor Random Forest achieves higher weighted F1 than a single-feature logistic regression on C-MAPSS FD001 (0.9576 vs 0.9247). But the logistic regression catches 98% of imminent failures versus 88% for the Random Forest. In predictive maintenance, that 10-point recall gap is the one that matters. Choose your metric before you train.

**2. Feature engineering often matters more than model complexity.** A rolling count of degraded-state cycles — a single calculated field — captures temporal failure dynamics that a snapshot-based model across 21 channels cannot easily learn.

**3. Domain framing transforms raw numbers into decisions.** The difference between "this sensor reads 487.2" and "this machine has spent 12 unhealthy intervals this week" is the difference between data and intelligence.

---

## 📁 Reproduce This Analysis

The full Jupyter Notebook for this study — using the publicly available NASA C-MAPSS turbofan engine dataset as a structural analogue — is available on GitHub:

🔗 **[github.com/Puru2001pandey/industrial-telemetry-analytics-research](https://github.com/Puru2001pandey/industrial-telemetry-analytics-research)**

The notebook covers all phases:
- Schema normalization & preprocessing
- EDA & failure rate comparison
- Custom downtime interval feature engineering
- **Phase 3B: Verified LR vs RF comparison** (reproducible — run cells, see the same 0.98 / 0.88 recall numbers)
- Forensic classification analysis (pay equity)

---

## 🔮 What's Next

This analysis opens several interesting research directions:

- **LSTM-based RUL Prediction** — instead of detecting failure after degradation begins, can we forecast remaining useful life 50+ cycles in advance?
- **Federated Learning** — how do we analyze multi-factory data without centralizing sensitive operational telemetry?
- **Multi-intersectional Forensic Analysis** — extending the equity study to analyze disparities across gender × department × location simultaneously

I'm actively working on the LSTM extension — if you're interested in collaborating or discussing the methodology, reach out on [LinkedIn](https://linkedin.com/in/puru-pandey-851271229).

---

*Puru Pandey is a B.Tech graduate in Artificial Intelligence & Machine Learning from Galgotias University. He builds data research tools, deployed analytics dashboards, and ML applications. His work includes the [India AQI Intelligence Dashboard](https://india-aqi-dashboard.streamlit.app) and the [Crop Yield Predictor](https://cropyieldprojectcrop.streamlit.app).*
