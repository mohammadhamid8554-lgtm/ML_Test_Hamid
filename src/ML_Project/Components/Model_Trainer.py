import sys
import os
from dataclasses import dataclass
from urllib.parse import urlparse

from catboost import CatBoostRegressor 

from sklearn.ensemble import ( 
    AdaBoostRegressor, 
    GradientBoostingRegressor,
    RandomForestRegressor

)

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor 
from xgboost import XGBRegressor 

from src.ML_Project.exception import CustomException
from src.ML_Project.logger import logging
from src.ML_Project.utils import save_obj, evaluate_models
import mlflow
import mlflow.sklearn as mlflow_sklearn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()



    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)

        return rmse, mae, r2



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


            print("This is the best model:")
            print(best_model_name)

            model_names = list(params.keys())

            actual_model = ""

            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model + model

            best_params =params[actual_model]

            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            registry_uri = os.getenv("MLFLOW_REGISTRY_URI")
            register_model = os.getenv("MLFLOW_REGISTER_MODEL", "").lower() in {
                "1",
                "true",
                "yes",
            }

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            if registry_uri:
                mlflow.set_registry_uri(registry_uri)

            # mlflow 

            with mlflow.start_run():

                predicted_qualities = best_model.predict(X_test)

                (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_params(best_params)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2",r2)
                mlflow.log_metric("mae", mae)

                if register_model and registry_uri:
                    mlflow_sklearn.log_model(
                        best_model, "Model", registered_model_name=actual_model
                    )
                else:
                    mlflow_sklearn.log_model(best_model, "Model")



            if best_model_score < 0.6:
                raise CustomException("No best model found", sys) # type: ignore
            logging.info("Best found model on both training and testing dataset")

            save_obj( file_path= self.model_trainer_config.trained_model_file_path, obj=best_model)

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square


        except Exception as e:
            raise CustomException(e, sys) # type: ignore
