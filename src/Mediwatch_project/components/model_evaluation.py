import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from Mediwatch_project.entity.config_entity import ModelEvaluationConfig
from Mediwatch_project.utils.common import save_json
from pathlib import Path


    