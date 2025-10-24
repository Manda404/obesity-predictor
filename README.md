---

## 🧠 ObesityPredictor

> *A complete, production-ready ML & MLOps project for obesity classification.*

---

### 🚀 Overview

**ObesityPredictor** is an end-to-end **Machine Learning + MLOps** project demonstrating the **full lifecycle of a predictive model** — from data preprocessing to model deployment and monitoring.

It classifies individuals into **obesity categories** (*Underweight, Normal, Overweight, Obese*) based on demographic and health indicators.

This project integrates:

* 🧩 **Multi-model competition** — CatBoost, XGBoost, LightGBM
* ⚙️ **MLflow tracking & Model Registry**
* 🧼 **Consistent preprocessing (train → inference)**
* 🔥 **Centralized logging via Loguru**
* 🧠 **FastAPI REST API for inference**
* 📊 **Streamlit dashboard for visualization**
* 🧪 **CI/CD & Docker orchestration**
* 📈 **Monitoring with Prometheus, Grafana & Evidently**

---

### 🧱 Project Structure

```
obesity-predictor/
│
├── pyproject.toml          # Poetry dependencies
├── README.md               # Project documentation
├── .env                    # Environment variables (MLflow, paths)
├── .gitignore
│
├── data/
│   ├── raw/                # Raw datasets
│   ├── processed/          # Processed datasets
│   └── models/             # Saved models & encoders
│
├── notebooks/              # Jupyter notebooks for exploration
│   ├── 01_eda.ipynb
│   └── 02_modeling_experiments.ipynb
│
├── obesity_predictor/      # Main source package
│   ├── config/             # Settings, logger, model configs
│   ├── core/               # Core ML logic
│   │   ├── data/           # Loading & splitting
│   │   ├── preprocessing/  # Train/inference preprocessors
│   │   ├── model/          # Trainers, evaluators, registry
│   │   ├── pipeline/       # Orchestration & inference pipelines
│   │   ├── utils/          # Helpers (MLflow, viz, serialization)
│   │   └── validation/     # Drift detection, schema validation
│   │
│   ├── api/                # FastAPI inference service
│   ├── app/                # Streamlit UI
│   └── ci_cd/              # Docker, GitHub Actions, monitoring configs
│
└── tests/                  # Unit & integration tests
```

---

### 🧩 Key Features

| Module                        | Description                                        |
| ----------------------------- | -------------------------------------------------- |
| **Data Handling**             | Robust data loading & stratified splitting         |
| **Feature Engineering**       | Consistent preprocessing across training/inference |
| **Model Trainers**            | CatBoost, XGBoost, LightGBM with unified API       |
| **Evaluation & Benchmarking** | Automated model comparison                         |
| **Experiment Tracking**       | MLflow for logging & model versioning              |
| **API Layer**                 | FastAPI for real-time prediction                   |
| **UI Layer**                  | Streamlit dashboard for visualization & insights   |
| **Monitoring**                | Evidently + Prometheus + Grafana                   |
| **Deployment**                | Docker + GitHub Actions (CI/CD ready)              |

---

### ⚙️ Installation

#### Requirements

* Python ≥ 3.10
* [Poetry](https://python-poetry.org/docs/#installation)
* (Optional) [Docker](https://docs.docker.com/get-docker/)

#### Setup

```bash
git clone https://github.com/Manda404/obesity-predictor.git
cd obesity-predictor
poetry install
```

---

### 🔧 Configuration

All configuration variables are in `.env`:

```env
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
EXPERIMENT_NAME=ObesityPredictor
MODEL_NAME=ObesityPredictor-Best
ARTIFACT_DIR=obesity_artifacts
TARGET_COLUMN=NObeyesdad
```

Start the MLflow UI locally:

```bash
poetry run mlflow ui --host 127.0.0.1 --port 5000
```

---

### 🧠 Training

Run the full training pipeline:

```bash
poetry run python -m obesity_predictor.core.pipeline.orchestration
```

This pipeline will:

* Load & preprocess the dataset
* Train **CatBoost**, **XGBoost**, **LightGBM** models
* Evaluate and compare their performance
* Log all metrics & artifacts in **MLflow**
* Register the best model

---

### ⚡ API (FastAPI)

Launch the inference API:

```bash
poetry run uvicorn obesity_predictor.api.main_api:app --reload
```

Test endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"Age": 25, "Gender": "Male", "Height": 175, "Weight": 70}'
```

---

### 📊 Streamlit Dashboard

Start the app:

```bash
poetry run streamlit run obesity_predictor/app/main_app.py
```

Pages:

* **Model Training Results**
* **Model Comparison**

---

### 🧪 Testing

```bash
poetry run pytest -v
```

---

### 🐳 Docker Deployment

Build & run with Docker Compose:

```bash
docker compose -f obesity_predictor/ci_cd/docker/docker-compose.yml up --build
```

Services included:

* FastAPI (inference)
* Streamlit (UI)
* MLflow (tracking)
* Prometheus / Grafana (monitoring)

---

### 📈 Monitoring

The project integrates:

* **Evidently** for data & concept drift detection
* **Prometheus** for metrics collection
* **Grafana** dashboards for visualization

---

### 🧰 Tech Stack

| Layer                   | Tools                                     |
| ----------------------- | ----------------------------------------- |
| **Core ML**             | Scikit-learn, CatBoost, XGBoost, LightGBM |
| **Experiment Tracking** | MLflow                                    |
| **Logging**             | Loguru                                    |
| **API**                 | FastAPI                                   |
| **UI**                  | Streamlit                                 |
| **Monitoring**          | Evidently, Prometheus, Grafana            |
| **Deployment**          | Docker, GitHub Actions                    |
| **Environment**         | Poetry                                    |

---

### 👨‍💻 Author

**Rostand Surel**
*ML Engineer & Data Scientist passionate about reliable, production-grade AI systems.*

📧 **[rostandsurel@yahoo.com](mailto:rostandsurel@yahoo.com)**
🌐 [LinkedIn](https://www.linkedin.com/in/rostand-surel/)
💻 [GitHub](https://github.com/Manda404)

---

### 🏁 Future Improvements

* 🔄 Add automated retraining (Airflow / Prefect)
* 💡 Model explainability (SHAP, LIME)
* 📬 Alert system for drift detection (Slack / Email)
* 🧪 Full test coverage + CI pipeline badges
* ☁️ Cloud deployment (AWS SageMaker / Azure ML)

---