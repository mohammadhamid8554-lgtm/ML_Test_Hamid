import sys
import os
from dataclasses import dataclass

from catboost import CatBoostRegressor # pyright: ignore[reportMissingImports]

from sklearn.ensemble import ( # pyright: ignore[reportMissingModuleSource]
    AdaBoostRegressor, 
    GradientBoostingRegressor,
    RandomForestRegressor

)

from sklearn.linear_model import LinearRegression # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import r2_score # pyright: ignore[reportMissingModuleSource]
from sklearn.neighbors import KNeighborsRegressor # pyright: ignore[reportMissingModuleSource]
from sklearn.tree import DecisionTreeRegressor # pyright: ignore[reportMissingModuleSource]
from xgboost import XGBRegressor # pyright: ignore[reportMissingImports]

from src.ML_Project.exception import CustomException
from src.ML_Project.logger import logging
from src.ML_Project.utils import save_obj, evaluate_models





@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:
            logging.info("Split training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "XGBoosting Regressor" : XGBRegressor(),
                "CatBoosting Regressor" : CatBoostRegressor(verbose = False),
                "AdaBoost Regressor" : AdaBoostRegressor()
            }

            params = {
                "Decision Tree" :{
                    "criterion": ["squared_error", "absolute_error", "poisson"],
                    # "spitters" : ["best", "random"],
                    # "max_features" : ["sqrt", "log2"]
                    },

                    "Random Forest":{ 
                        # "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisso"],
                        # "max_features" : ["sqrt", "log2", None],
                        "n_estimators": [8, 16, 32, 64, 128, 256]
                    },

                    "Gradient Boosting":{

                        # "Loss" : ["squared_error", "friedman_mse", "absolute_error", "poisso"],
                        # "criteion":["squared_error", "friedman_mse"],
                        "n_estimators": [8, 16, 32, 64, 128, 256],
                       "learning_rate" : [.1, .01,.05, .001],
                       "subsample" : [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]

                    },

                    "Linear Regression" : {},
                          "XGBoosting Regressor": {
                              "learning_rate" : [.1, .01,.05, .001],
                       "n_estimators": [8, 16, 32, 64, 128, 256]

                    },

                    "CatBoosting Regressor" :{
                        "depth" : [6, 8, 10],
                        "learning_rate" : [.1, .01,.05, .001],
                        "n_estimators": [8, 16, 32, 64, 128, 256]


                    },

                    "AdaBoost Regressor" : {
                                 "learning_rate" : [.1, .01,.05, .001],
                                 "n_estimators": [8, 16, 32, 64, 128, 256],
                        # "Loss" : ["square", "linear", "exponential"],

                    }
            }

            model_report: dict = evaluate_models(
                X_train, X_test, y_train, y_test, models, params
            )

            # To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys) # type: ignore
            logging.info("Best found model on both training and testing dataset")

            save_obj( file_path= self.model_trainer_config.trained_model_file_path, obj=best_model)

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square


        except Exception as e:
            raise CustomException(e, sys) # type: ignore
