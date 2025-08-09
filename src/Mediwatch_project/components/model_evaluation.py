import os
import pandas as pd
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from Mediwatch_project.entity.config_entity import ModelEvaluationConfig
from Mediwatch_project.utils.common import save_json
from pathlib import Path
import json
from Mediwatch_project import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self,actual, pred):
        acc = accuracy_score(actual, pred)
        prec = precision_score(actual, pred, average="binary",zero_division=0)
        rec = recall_score(actual, pred, average="binary",zero_division=0)
        f1 = f1_score(actual, pred, average="binary",zero_division=0)
        return acc, prec, rec, f1    

    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        with mlflow.start_run():

            predicted_qualities = model.predict(test_x)

            acc, prec, rec, f1 = self.eval_metrics(test_y, predicted_qualities)
            
            scores = {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "n_samples": int(len(test_y)),
                "positive_rate": float(np.mean(test_y == 1)),
            }

            out_path = Path(self.config.metric_file_name)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(scores, f, indent=2)
            

            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)


            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="XGB Classifier")
            else:
                mlflow.sklearn.log_model(model, "model")
    