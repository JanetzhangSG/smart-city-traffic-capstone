# Part 2 – Python Traffic Analytics Pipeline

This folder contains the final verified submission for Part 2 of the Smart City Traffic Intelligence capstone.

## Contents

- `data/raw/` – original Metro Interstate Traffic Volume CSV used by the pipeline.
- `pipeline.py` – data loading, schema validation, cleaning, outlier handling and logging.
- `feature_engineering.py` – feature engineering for time, weather and congestion analysis.
- `visualizations.py` – Matplotlib visualisations and analytical outputs.
- `cli_app/` – command-line application for interacting with the processed traffic data.
- `outputs/` – cleaned and feature-engineered datasets.
- `figures/` – generated traffic visualisations.
- `logs/` – verified pipeline and application logs.
- `report/` – final Part 2 methodology and findings report.
- `requirements.txt` – Python dependencies.

## Reproduce

Run these commands from the `part2_python` directory.

### 1. Install dependencies

```bash
pip install -r requirements.txt
