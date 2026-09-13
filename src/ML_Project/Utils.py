import os
from src.ML_Project.exception import CustomException
from src.ML_Project.logger import logging
import pandas as pd
from dataclasses import dataclass
import sys
from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
import pickle
from sklearn.model_selection import GridSearchCV # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import r2_score # pyright: ignore[reportMissingModuleSource]




load_dotenv()

def read_sql_data():
    logging.info("Reading SQL Database Started!!")

    try:
        import pymysql # pyright: ignore[reportMissingModuleSource]

        host = os.getenv("host")
        user = os.getenv("user")
        password = os.getenv("password")
        db = os.getenv("db")
        if password is None:
            raise ValueError("The 'password' environment variable is not set")

        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password, # type: ignore
            db=db
            )
        logging.info("Connection Establish!!")
        df = pd.read_sql_query("select * from students", mydb)
        print(df.head())
        return df
        
    except Exception as ex:
        raise CustomException(ex,sys) # type: ignore


def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys) # type: ignore


def evaluate_models(X_train, X_test, y_train, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Model.fit(X_train, y_train) --> Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys) # type: ignore

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException (e, sys) # type: ignore



