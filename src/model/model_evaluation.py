import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import dagshub
import mlflow.lightgbm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature

# -----------------------------
# Logging Configuration
# -----------------------------
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_evaluation_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# -----------------------------
# Utility Functions
# -----------------------------
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.fillna("", inplace=True)
    return df


def load_model(model_path: str):
    with open(model_path, "rb") as file:
        return pickle.load(file)



def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    with open(vectorizer_path, "rb") as file:
        return pickle.load(file)


def load_params(params_path: str) -> dict:
    with open(params_path, "r") as file:
        return yaml.safe_load(file)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return report, cm


def log_confusion_matrix(cm, dataset_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix for {dataset_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_file_path = f"confusion_matrix_{dataset_name}.png"
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()


def save_model_info(run_id: str, file_path: str):
    model_info = {
        "run_id": run_id,
        "model_path": "model"  # IMPORTANT: must match log_model artifact_path
    }
    with open(file_path, "w") as file:
        json.dump(model_info, file, indent=4)


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():

    # Initialize MLflow + DagsHub
    dagshub.init(
        repo_owner="prasu202324",
        repo_name="YTintel-extension",
        mlflow=True
    )

    mlflow.set_tracking_uri(
        "https://dagshub.com/prasu202324/YTintel-extension.mlflow/"
    )

    mlflow.set_experiment("dvc-pipeline-runs")

    with mlflow.start_run() as run:

        root_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../")
        )

        # -----------------------------
        # Load Params
        # -----------------------------
        params = load_params(os.path.join(root_dir, "params.yaml"))
        for key, value in params.items():
            mlflow.log_param(key, value)

        # -----------------------------
        # Load Model & Vectorizer
        # -----------------------------
        model = load_model(os.path.join(root_dir, "lgbm_model.pkl"))
        vectorizer = load_vectorizer(
            os.path.join(root_dir, "tfidf_vectorizer.pkl")
        )

        # -----------------------------
        # Load Test Data
        # -----------------------------
        test_data = load_data(
            os.path.join(root_dir, "data/interim/test_processed.csv")
        )

        X_test_tfidf = vectorizer.transform(
            test_data["clean_comment"].values
        )

        y_test = test_data["category"].values

        # -----------------------------
        # Signature Inference
        # -----------------------------
        input_example = pd.DataFrame(
            X_test_tfidf.toarray()[:5],
            columns=vectorizer.get_feature_names_out()
        )

        signature = infer_signature(
            input_example,
            model.predict(X_test_tfidf[:5])
        )

        # -----------------------------
        # 🔥 LOG MODEL (CORRECT WAY)
        # -----------------------------
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",   # STANDARD NAME
            signature=signature,
            input_example=input_example
        )

        # Save run info for registry step
        save_model_info(run.info.run_id, "experiment_info.json")

        # Log vectorizer
        mlflow.log_artifact(
            os.path.join(root_dir, "tfidf_vectorizer.pkl")
        )

        # -----------------------------
        # Evaluation
        # -----------------------------
        report, cm = evaluate_model(model, X_test_tfidf, y_test)

        for label, metrics in report.items():
            if isinstance(metrics, dict):
                mlflow.log_metrics({
                    f"test_{label}_precision": metrics["precision"],
                    f"test_{label}_recall": metrics["recall"],
                    f"test_{label}_f1-score": metrics["f1-score"],
                })

        log_confusion_matrix(cm, "Test_Data")

        # Tags
        mlflow.set_tag("model_type", "LightGBM")
        mlflow.set_tag("task", "Sentiment Analysis")
        mlflow.set_tag("dataset", "YouTube Comments")

        print("✅ Run completed successfully!")
        print("Run ID:", run.info.run_id)


if __name__ == "__main__":
    main()