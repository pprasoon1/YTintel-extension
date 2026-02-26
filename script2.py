import mlflow
import dagshub
from mlflow.tracking import MlflowClient

dagshub.init(repo_owner='prasu202324', repo_name='YTintel-extension', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/prasu202324/YTintel-extension.mlflow/")
client = MlflowClient()

client.transition_model_version_stage(
    name="yt_chrome_plugin_model",
    version=5,   
    stage="Production"
)

print("Version 5 moved to Production")