import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import Custon_exception
from src.utils import save_path, evaluate_model

@dataclass
class Model_train_config:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class Model_train:
    def __init__(self):
        self.model_train_config = Model_train_config()

    def initiate_model_train(self, train_arr, test_arr):
        try:
            logging.info("Model training started")
            logging.info("Splitting train and test data")
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1], 
                train_arr[:, -1], 
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                "random_forest": RandomForestRegressor(),
                "decision_tree": DecisionTreeRegressor(),
                "linear_regression": LinearRegression(),
                "cat_boost": CatBoostRegressor(verbose=False),
                "k-neighbour": KNeighborsRegressor(),
                "xgboost": XGBRegressor(),
                "gradient_boost": GradientBoostingRegressor()
            }

            model_report = evaluate_model(
                x_train=x_train, y_train=y_train,
                x_test=x_test, y_test=y_test,
                models=models
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise Custon_exception("No suitable model found")

            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")

            save_path(
                file_path=self.model_train_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2_sq = r2_score(y_test, predicted)
            return r2_sq

        except Exception as e:
            raise Custon_exception(e, sys)
