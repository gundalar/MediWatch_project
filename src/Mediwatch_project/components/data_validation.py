import os
from Mediwatch_project import logger
from Mediwatch_project.entity.config_entity import DataValidationConfig
import pandas as pd
import yaml
from box import Box


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config


    def validate_all_columns(self)-> bool:
        try:
            validation_status = None

            with open("schema.yaml", "r") as file:
                schema = Box(yaml.safe_load(file))

            # Access columns
            columns = schema.COLUMNS
            target_column = schema.target_column.name

            data = pd.read_csv(self.config.unzip_data_dir)

            def validate_schema(df, schema):
                missing_columns = [col for col in schema.COLUMNS if col not in df.columns]
                if missing_columns:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                    raise ValueError(f"Missing columns: {missing_columns}")

                for col, expected_type in schema.COLUMNS.items():
                    if col in df.columns:
                        if df[col].dtype.name != expected_type:
                            print(f"Warning: Column '{col}' has type {df[col].dtype}, expected {expected_type}")
                
                
                validation_status = True
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation status: {validation_status}")
                return validation_status

            validation_status = validate_schema(data, schema)

            return validation_status
        
        except Exception as e:
            raise e
