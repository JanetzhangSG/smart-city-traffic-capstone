# Part 2 – Python Traffic Analytics Pipeline

This folder contains the final verified submission for Part 2 of the Smart City Traffic Intelligence capstone.

## Contents
- `00_Data/raw/` – original Metro Interstate Traffic Volume CSV used by the pipeline.
- `01_Python_Scripts/` – `pipeline.py`, `feature_engineering.py`, `visualizations.py`, and `app.py`.
- `02_Outputs/` – cleaned and feature-engineered datasets.
- `03_Figures/` – three Matplotlib visualisations required for Task 3.
- `04_Logging/` – verified `pipeline.log` showing pipeline, feature-engineering, visualisation, and CLI events.
- `05_Report/` – final Part 2 methodology and findings report (PDF).
- `requirements.txt` – Python dependencies.

## Reproduce
Run these commands from the `01_Python_Scripts` directory, with the relative paths adjusted as shown:

```bash
pip install -r ../requirements.txt
python pipeline.py
python feature_engineering.py --input ../02_Outputs/traffic_cleaned.csv --output ../02_Outputs/traffic_features.csv --log ../04_Logging/pipeline.log --log-level DEBUG
python visualizations.py --input ../02_Outputs/traffic_features.csv --output-dir ../03_Figures --log ../04_Logging/pipeline.log --log-level INFO
```

Example CLI commands:

```bash
python app.py hourly 16
python app.py compare
python app.py recommend
python app.py weather Clear
python app.py date 2017-08-01
```

## Verified Results
The verified run loaded **48,204 rows × 9 columns**, removed **17 duplicate rows**, handled **10** physically implausible 0 K temperature readings and **120** rainfall outliers, and produced **48,187 cleaned rows**.

Feature engineering produced **48,187 rows × 36 columns**. The main visual findings were a traffic peak around **16:00**, lower average traffic on weekends than weekdays, and a weak positive temperature/traffic relationship (approximately **r = 0.1323** after cleaning).

## Logging
Python's `logging` module is used throughout the project. The log records:
- `DEBUG` – intermediate calculations such as scaling ranges and congestion thresholds.
- `INFO` – successful loading, processing milestones, saved datasets, saved figures, and CLI commands.
- `WARNING` – recoverable data-quality changes such as duplicate removal and imputation.
- `ERROR` – failures or invalid CLI input.

The verified log is stored in `04_Logging/pipeline.log`.

## Note on the Colab notebook
The Colab `.ipynb` used during development is kept as a separate personal backup and is not part of this final submission package. The required reproducible implementation is provided as Python scripts.
