import os
from src.ML_Project.exception import CustomException
from src.ML_Project.logger import logging
import pandas as pd
from dataclasses import dataclass
import sys
from dotenv import load_dotenv
import pymysql




load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

if password is None:
    raise ValueError("The 'password' environment variable is not set")

def read_sql_data():
    logging.info("Reading SQL Database Started!!")

    try:
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
    


