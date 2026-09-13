import os
from src.ML_Project.exception import CustomException
from src.ML_Project.logger import logging
import pandas as pd
from dataclasses import dataclass
import sys
from dotenv import load_dotenv
import pickle




load_dotenv()

def read_sql_data():
    logging.info("Reading SQL Database Started!!")

    try:
        import pymysql

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
        raise CustomException(e, sys)
    


