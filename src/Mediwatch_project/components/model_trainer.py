import pandas as pd
import os
from Mediwatch_project import logger
import joblib
import xgboost as xgb
import numpy as np
import sklearn
from Mediwatch_project.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Load training and testing data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # Split features and target
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        # Handle categorical columns
        for col in train_x.select_dtypes(['category']).columns:
            train_x[col] = train_x[col].cat.codes
            test_x[col] = test_x[col].cat.codes

        # Handle class imbalance
        neg, pos = np.bincount(train_y)
        scale_pos_weight = neg / pos

        # Define XGBoost classifier using parameters from config
        xgb_clf = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate
        )

        # Train the model
        xgb_clf.fit(train_x, train_y)

        # Save the trained model
        joblib.dump(xgb_clf, os.path.join(self.config.root_dir, self.config.model_name))