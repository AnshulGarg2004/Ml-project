import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import Custon_exception
from src.logger import logging
from src.utils import save_path

class Data_tranformation_config:
    preprocessor_obj_file_path = os.path.join('artifacts', 'proprocessor.pkl')

class Data_transformation:
    def __init__(self):
        self.data_trans_config = Data_tranformation_config()

    def get_data_trans_obj(self):
        '''
        Trhis function is responsible for data transformation
        '''
        try:
            numerical_feat = ['writing_score', 'reading_score']
            categorical_feat = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            num_pipeline = Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("standard_scaling", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical columns standard scaling done")
            logging.info("Categorical columns encoding completed")
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_feat),
                    ('cat_pipeline', cat_pipeline, categorical_feat)
                ]
            )

            return preprocessor
        except Exception as e:
            raise Custon_exception(e, sys)

    def inittiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("btaining preprocessing object")

            preprocessing_obj = self.get_data_trans_obj()

            target_column_name = "math_score"
            numerical_feat = ['writing_score', 'reading_score']

            input_feat_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feat_traaain_df = train_df[target_column_name]

            input_feat_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feat_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and test dataframe")
            input_feat_train_arr = preprocessing_obj.fit_transform(input_feat_train_df)
            input_deat_test_arr = preprocessing_obj.transform(input_feat_test_df)

            train_arr = np.c_[
                input_feat_train_arr, np.array(target_feat_traaain_df)
            ]
            test_arr = np.c_[input_deat_test_arr, np.array(target_feat_test_df)]
            logging.info("Saved processing object")

            save_path(
                file_path = self.data_trans_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr, test_arr, self.data_trans_config.preprocessor_obj_file_path
            )
        except  Exception as e:
            raise Custon_exception(e, sys)

    
            