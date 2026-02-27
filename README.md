# 🚀 YTintel: YouTube Sentiment Insights & MLOps Pipeline

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95.1-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML_Library-F7931E.svg)](https://scikit-learn.org/)
[![DVC](https://img.shields.io/badge/DVC-Data_Version_Control-945dd6.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED.svg)](https://www.docker.com/)
[![DagsHub](https://img.shields.io/badge/DagsHub-MLops_Platform-00B894.svg)](https://dagshub.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**YTintel** is a powerful End-to-End MLOps project that provides real-time sentiment analysis of YouTube comments. It combines a sleek Chrome extension frontend with a robust FastAPI backend, all managed through a professional data science lifecycle including automated pipelines, model versioning, and experiment tracking.

---

## 📺 Project Overivew

| extension_popup | analysis_results |
|:---:|:---:|
| ![Extension UI](./assets/extension_ui.png) | ![Analysis Results](./assets/analysis_results.png) |

> [!TIP]
> **YTintel** doesn't just predict sentiment; it provides high-level insights like **Sentiment Trends Over Time**, **Word Clouds**, and **User Engagement Metrics** directly in your browser.

---

## 🛠 Tech Stack

### 🧠 Machine Learning & MLOps
- **LightGBM**: Gradient boosting framework for high-performance sentiment classification.
- **DVC (Data Version Control)**: Orchestrates the data engineering and model training pipeline.
- **MLflow**: Tracks experiments, metrics, and manages the model registry.
- **DagsHub**: Cloud-based storage for DVC remote and MLflow tracking.
- **Scikit-learn**: TF-IDF vectorization and evaluation utilities.

### 🌐 Backend & DevOps
- **FastAPI**: High-performance asynchronous API for real-time predictions.
- **Docker & Docker Compose**: Containerization for seamless deployment.
- **GitHub Actions**: CI/CD pipeline for automated linting and container builds.
- **Matplotlib/WordCloud**: Dynamic visualization generation on-demand.

### 🧩 Frontend
- **Chrome Extension (Manifest V3)**: Native browser integration.
- **YouTube Data API v3**: Efficient comment scraping.

---

## 🏗 Methodology: The MLOps Lifecycle

The project follows a rigorous engineering methodology to ensure model reproducibility and reliable deployment.

### 1. Data Pipeline (DVC)
Our entire workflow is modularized into stages defined in `dvc.yaml`:
- **Ingestion**: Automated fetching and splitting of training/test data.
- **Preprocessing**: Robust NLP cleaning (Lemmatization, stop-word removal, TF-IDF).
- **Training**: Optimized LightGBM model training with hyperparameter tracking.
- **Evaluation**: Comprehensive metric logging (Accuracy, Precision, Recall).

### 2. Experiment Tracking (MLflow)
Every run is logged to **MLflow** on DagsHub, capturing:
- **Parameters**: `n_estimators`, `max_depth`, `learning_rate`.
- **Metrics**: Validation performance.
- **Artifacts**: The serialized model (`lgbm_model.pkl`) and TF-IDF vectorizer.

---

## 📊 Performance & Insights

We achieve high-fidelity sentiment classification by addressing class imbalance and fine-tuning feature extraction.

### Model Evaluation
| distribution_chart | word_cloud |
|:---:|:---:|
| ![Sentiment Trend](./assets/trend_graph.png) | ![Word Cloud](./assets/word_cloud.png) |

The model is evaluated using a detailed confusion matrix to ensure balanced performance across **Positive**, **Neutral**, and **Negative** sentiments.

---

## 🚀 Getting Started

### Backend (Docker - Recommended)
```bash
# Clone the repository
git clone https://github.com/pprasoon1/YTintel-extension.git
cd YTintel-extension

# Start the API via Docker Compose
docker-compose up --build
```
The API serves at `http://localhost:8000`.

### Frontend (Chrome Extension)
1. Open Chrome and navigate to `chrome://extensions/`.
2. Enable **Developer mode**.
3. Click **Load unpacked**.
4. Select the `yt-chrome-plugin-frontend` folder.
5. Pin the extension and start analyzing!

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
Created with ❤️ by [Prasoon](https://github.com/pprasoon1)
