import os
import sys
import pandas as pd
import numpy as np
import re # Import the regular expression module
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.logging.logger import logging
from src.exception.exception import CustomException
from src.utils.common import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('models', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # These feature names match the output of our new, robust cleaning function
            categorical_features = ['crop', 'state']
            numerical_features = [
                'cost_of_cultivation_hectare_a2_fl',
                'cost_of_cultivation_hectare_c2',
                'cost_of_production_quintal_c2'
            ]

            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', numerical_pipeline, numerical_features),
                    ('cat_pipeline', categorical_pipeline, categorical_features)
                ],
                remainder='passthrough'
            )
            
            logging.info("Preprocessor object created successfully.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            # --- THE DEFINITIVE FIX: Robust Column Cleaning and Preparation ---
            def clean_and_prepare_df(df):
                logging.info(f"Original column names: {df.columns.tolist()}")

                cleaned_columns = []
                for col in df.columns:
                    # 1. Strip whitespace and convert to lowercase
                    new_col = col.strip().lower()
                    # 2. Use regex to replace all non-alphanumeric characters with a single underscore
                    new_col = re.sub(r'[^a-z0-9]+', '_', new_col)
                    # 3. Remove any leading/trailing underscores that might result
                    new_col = new_col.strip('_')
                    cleaned_columns.append(new_col)
                
                df.columns = cleaned_columns
                logging.info(f"Cleaned with regex: {df.columns.tolist()}")

                # 4. Explicitly find the column containing 'yield' and rename it to 'yield'
                # The original "Yield (Quintal/ Hectare) " will become something like "yield_quintal_hectare"
                try:
                    yield_col_name = [col for col in df.columns if 'yield' in col][0]
                    df.rename(columns={yield_col_name: 'yield'}, inplace=True)
                    logging.info(f"Final column names after renaming yield: {df.columns.tolist()}")
                except IndexError:
                    raise KeyError("FATAL: No column containing the word 'yield' was found in the data. Please check your CSV file.")

                return df

            train_df = clean_and_prepare_df(train_df)
            test_df = clean_and_prepare_df(test_df)

            target_column_name = 'yield'
            
            # This check now happens AFTER the 'yield' column has been created and verified
            if target_column_name not in train_df.columns:
                raise KeyError(f"Target column '{target_column_name}' not found after cleaning. This should not happen.")

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object.")
            
            preprocessor_obj = self.get_data_transformer_object()
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info("Saved new preprocessing object.")

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)
        