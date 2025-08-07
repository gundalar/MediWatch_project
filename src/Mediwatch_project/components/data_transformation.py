import os
from Mediwatch_project import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from Mediwatch_project.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_spliting(self):
        df = pd.read_csv(self.config.data_path)

        # Step 1: Basic Cleaning and Encoding
        df = df[df['gender'] != 'Unknown/Invalid']
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

        df['race'] = df['race'].replace('?', 'Other')
        age_map = {f'[{10*i}-{10*(i+1)})': i for i in range(10)}
        df['age'] = df['age'].map(age_map)

        for col in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']:
            df[col] = df[col].fillna(0).astype(int)

        df['change'] = df['change'].map({'No': 0, 'Ch': 1, 'Yes': 1}).fillna(0)
        df['diabetesMed'] = df['diabetesMed'].map({'No': 0, 'Yes': 1}).fillna(0)

        medication_cols = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'insulin', 'glyburide-metformin','glipizide-metformin', 'glimepiride-pioglitazone',
            'metformin-rosiglitazone','metformin-pioglitazone']

        med_map = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}
        for col in medication_cols:
            if col in df.columns:
                df[col] = df[col].map(med_map).fillna(0).astype(int)
            else:
                print(f"Warning: Column '{col}' not found in dataset.")

        readmitted_mapping = {'NO': 0, '>30': 0, '<30': 1}
        df['readmitted'] = df['readmitted'].map(readmitted_mapping)

        # Step 2: Diagnosis Mapping
        def map_diag(code):
            try:
                code = str(code)
                if code.startswith('V') or code.startswith('E'):
                    return 'Other'
                code = float(code)
                if 390 <= code <= 459 or code == 785:
                    return 'Circulatory'
                elif 460 <= code <= 519 or code == 786:
                    return 'Respiratory'
                elif 520 <= code <= 579 or code == 787:
                    return 'Digestive'
                elif 250 <= code < 251:
                    return 'Diabetes'
                elif 800 <= code <= 999:
                    return 'Injury'
                elif 710 <= code <= 739:
                    return 'Musculoskeletal'
                elif 580 <= code <= 629 or code == 788:
                    return 'Genitourinary'
                elif 140 <= code <= 239:
                    return 'Neoplasms'
                else:
                    return 'Other'
            except:
                return 'Unknown'

        df['diag_1_cat'] = df['diag_1'].apply(map_diag)
        df['diag_2_cat'] = df['diag_2'].apply(map_diag)
        df['diag_3_cat'] = df['diag_3'].apply(map_diag)

        # Step 3: Drop columns
        cols_to_drop = ['encounter_id', 'patient_nbr', 'payer_code', 'weight', 'medical_specialty',
                        'max_glu_serum', 'A1Cresult', 'examide', 'citoglipton',
                        'diag_1', 'diag_2', 'diag_3']
        df = df.drop(columns=cols_to_drop, errors='ignore')

        df['gender'] = df['gender'].astype('category')
        df['readmitted'] = df['readmitted'].astype('category')

        # Step 4: One-hot encoding
        one_hot_encode_columns = [
            'race', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'insulin', 'glyburide-metformin','glipizide-metformin', 'glimepiride-pioglitazone',
            'metformin-rosiglitazone','metformin-pioglitazone', 'change', 'diabetesMed',
            'diag_1_cat', 'diag_2_cat', 'diag_3_cat']

        df_encoded = pd.get_dummies(df, columns=one_hot_encode_columns, drop_first=True, dtype=int)

        # Step 5: Train-test split (80/20)
        train, test = train_test_split(df_encoded, test_size=0.2, random_state=42)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Split data into training and test sets")
        logger.info(f"Train shape: {train.shape}")
        logger.info(f"Test shape: {test.shape}")

        print("Train shape:", train.shape)
        print("Test shape:", test.shape)