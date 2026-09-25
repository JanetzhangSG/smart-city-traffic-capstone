# Smart City Traffic Intelligence Capstone

This repository is the final portfolio submission for Parts 1, 2 and 3 of the Smart City Traffic Intelligence capstone.

## Repository structure

```text
smart-city-traffic-capstone/
|
|-- part1_data_analytics/
|   |-- sql/
|   |-- powerbi/
|   |-- statistics/
|   `-- insights_report.pdf
|
|-- part2_python/
|   |-- pipeline.py
|   |-- feature_engineering.py
|   |-- visualizations.py
|   |-- cli_app/
|   |-- figures/
|   |-- outputs/
|   |-- logs/
|   |-- report/
|   |-- data/raw/
|   |-- README.md
|   `-- requirements.txt
|
|-- part3_machine_learning/
|   |-- notebooks/
|   |-- models/
|   |-- mlflow/
|   |-- deployment/
|   |-- recommendation_system/
|   |-- monitoring/
|   |-- responsible_ai_report.pdf
|   |-- final_capstone_report.pdf
|   |-- README.md
|   `-- requirements.txt
|
|-- README.md
`-- requirements.txt
```

The Part 3 layout follows the suggested repository structure in the assignment. Task 1-3 scripts and their supporting model artefacts/results are centralised under `models/`; deployment, recommendation, monitoring and MLflow files are kept in the corresponding functional folders.

## Part 1 - Data Analytics

Includes the SQLite query file and evidence, statistical and probability analysis and evidence, the Power BI Desktop dashboard and evidence, and the required Data Analytics Insights Report.

## Part 2 - Python Traffic Analytics Pipeline

Includes the reproducible Python cleaning pipeline, feature engineering, Matplotlib visualisations, CLI application, generated outputs, logging, report, raw input data, README and requirements.

## Part 3 - Machine Learning and AI

Includes supervised and unsupervised learning, deep learning and SHAP explainability, MLflow experiment tracking, traffic timing recommendations, FastAPI deployment simulation, monitoring, model versioning, the final execution notebook and responsible-AI reporting.

### Important accident-data limitation

No accident dataset was supplied for the capstone. The Part 3 classification target `high_risk` is therefore a documented proxy based on high/severe congestion occurring together with severe or low-visibility weather. It is used only to demonstrate the classification workflow and must not be interpreted as a prediction of actual accidents.

## Reproducibility

Part 2 and Part 3 contain their own README and requirements files. The Part 3 notebook `part3_machine_learning/notebooks/Capstone_Part3_Final_Execution.ipynb` records the clean execution of Tasks 1-6. Task 7 is supplied as `part3_machine_learning/responsible_ai_report.pdf`.

Results are specific to the supplied single-corridor traffic dataset and should not be generalised beyond its coverage without further validation.
