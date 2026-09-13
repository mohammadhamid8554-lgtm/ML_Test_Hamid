# 1. Load Data From MySQL
# 2. Split Data Into Train and Test Data Sets

import os
from src.ML_Project.exception import CustomException
from src.ML_Project.logger import logging
import pandas as pd
from dataclasses import dataclass
import sys
from src.ML_Project.utils import read_sql_data
from sklearn.model_selection import train_test_split # type: ignore

# Data Class

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts", "train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")
    raw_data_path:str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        

    def initiate_data_ingestion(self):
        try:
            ## Reading data from MySQL
            df = pd.read_csv(os.path.join("notebook","raw.csv"))
            logging.info("Reading from SQL Database completed!!")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok= True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion Completed!!")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            




        except CustomException:
            raise
        except Exception as e:
            raise CustomException(e, sys) # type: ignore

