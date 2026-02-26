import mlflow
import dagshub
from mlflow.tracking import MlflowClient

dagshub.init(repo_owner='prasu202324', repo_name='YTintel-extension', mlflow=True)

mlflow.set_tracking_uri(
    "https://dagshub.com/prasu202324/YTintel-extension.mlflow/"
)

MODEL_NAME = "yt_chrome_plugin_model"
RUN_ID = "1dc6148a35d545c7bf52eb40fa49efdb"   # your latest run

client = MlflowClient()

# 🔥 THIS IS THE IMPORTANT PART
artifact_uri = mlflow.get_run(RUN_ID).info.artifact_uri + "/model"

model_version = client.create_model_version(
    name=MODEL_NAME,
    source=artifact_uri,
    run_id=RUN_ID
)

print("Registered Version:", model_version.version)

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=model_version.version,
    stage="Production"
)

print("Moved to Production")