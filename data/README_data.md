# Data Sources

## Primary Dataset: NASA C-MAPSS

- **Full Name:** Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)
- **Source:** NASA Intelligent Systems Division, Ames Research Center
- **Download:** https://www.nasa.gov/intelligent-systems-division/ (search "CMAPSS")
- **License:** Public domain (U.S. government dataset)

## Files Required

Place the following files in this `/data/` directory:

```
data/
├── train_FD001.txt    ← Single operating condition, one failure mode
├── train_FD002.txt    ← Six operating conditions, one failure mode
├── train_FD003.txt    ← Single operating condition, two failure modes
├── train_FD004.txt    ← Six operating conditions, two failure modes
├── test_FD001.txt
├── test_FD002.txt
├── test_FD003.txt
├── test_FD004.txt
└── RUL_FD001.txt      ← Ground truth Remaining Useful Life
```

## Structural Analogy to Original Study

| NASA CMAPSS Property | Original Study Analogue |
|---|---|
| Engine unit ID | Machine asset tag / machine_id |
| Operational cycle | Timestamp / telemetry tick |
| 3 operating settings | Operating mode / shift type |
| 21 sensor readings | IoT sensor stream values |
| RUL (remaining useful life) | Time-to-failure signal |
| Multiple datasets (FD001–FD004) | Multiple factory sites |

## Citation

Saxena, A. and Goebel, K. (2008).
"Turbofan Engine Degradation Simulation Data Set",
NASA Ames Prognostics Data Repository,
NASA Ames Research Center, Moffett Field, CA.

> **Note:** The original forensic study dataset is confidential and cannot
> be shared. This public dataset is used to demonstrate the analytical
> methodology on structurally equivalent industrial telemetry data.
