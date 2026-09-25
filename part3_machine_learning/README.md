# Part 3 - Machine Learning and AI

This folder contains the final Part 3 submission for the Smart City Traffic Intelligence capstone.

## Important accident-data limitation

No accident dataset was supplied for this capstone. The Task 1 classification target `high_risk` is therefore a documented proxy based on high/severe congestion occurring together with severe or low-visibility weather. The proxy is used only to demonstrate the machine-learning classification workflow and must not be interpreted as a prediction of actual accidents.

## Project structure

```text
part3_machine_learning/
|
|-- notebooks/
|   `-- Capstone_Part3_Final_Execution.ipynb
|
|-- models/
|   |-- Task 1-3 Python scripts
|   |-- trained model files (.joblib)
|   |-- Task 1-3 metrics/results
|   |-- Task 1-3 figures
|   `-- Task 1-3 execution logs
|
|-- mlflow/
|   |-- task4_mlflow.py
|   |-- task4_mlflow_runs.csv
|   |-- task4_mlflow_summary.json
|   |-- task4_mlflow.log
|   `-- README.md
|
|-- deployment/
|   |-- task6_api.py
|   |-- task6_api_test.py
|   |-- task6_api.log
|   `-- DEPLOYMENT_NOTE.md
|
|-- recommendation_system/
|   |-- task5_recommendation.py
|   |-- recommendation outputs
|   |-- task5_recommended_windows.png
|   `-- task5.log
|
|-- monitoring/
|   |-- task6_mlop.py
|   |-- task6_model_version_registry.csv
|   |-- task6_mlop_summary.json
|   |-- task6_monitoring_report.json
|   `-- task6.log
|
|-- responsible_ai_report.pdf
|-- final_capstone_report.pdf
|-- requirements.txt
`-- README.md
```

The cleaned traffic dataset used by Part 3 is produced in Part 2 and is intentionally not duplicated here in the final portfolio.

## Tasks covered

**Task 1 - Supervised Machine Learning**  
Classification uses Logistic Regression and Random Forest. Regression uses Linear Regression and HistGradientBoosting. The required evaluation metrics are recorded in the model results.

**Task 2 - Unsupervised Machine Learning**  
K-means clustering identifies traffic-condition groups using a data-driven selection of k. Association-rule mining identifies combinations of time, day type and weather associated with congestion levels.

**Task 3 - Deep Learning with Explainability**  
An MLP neural network predicts traffic volume. SHAP is applied to a comparable Random Forest trained on the same features and target because tree-based SHAP is more directly interpretable for this task.

**Task 4 - Advanced AI: MLflow**  
MLflow records model experiments, versions, metrics, metadata and model artifacts using a local SQLite tracking backend.

**Task 5 - Traffic Recommendation System**  
Because the data represents one corridor, the recommendation system focuses on travel timing rather than alternative routes. It supports day-type and weather filters and produces plain-language travel windows.

**Task 6 - MLOps and Deployment Simulation**  
The project documents model versions, extends MLflow tracking, serves a traffic-volume model through a FastAPI mock deployment, monitors feature/prediction drift and reports PASS/ALERT status.

**Task 7 - Responsible and Sustainable AI**  
`responsible_ai_report.pdf` covers coverage limitations, proxy-label risks, possible uneven errors, governance, oversight and sustainability/resource trade-offs.

## Reproducibility

The primary execution record is `notebooks/Capstone_Part3_Final_Execution.ipynb`, which contains the clean sequential execution of Tasks 1-6. The final reports contain the methodology, findings and limitations across the completed tasks.

The Part 3 scripts use Python logging with `logging.getLogger(__name__)` for internal status and progress reporting. The Task 6 monitoring thresholds are illustrative simulation thresholds rather than production policy.
