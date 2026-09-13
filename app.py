from src.ML_Project.logger import logging
from src.ML_Project.exception import CustomException
import sys
from src.ML_Project.components.Data_Ingestion import DataIngestion
from src.ML_Project.components.Data_Tranformation import DataTranformationConfig,DataTransformation


if __name__ == "__main__":
    logging.info("The execution has started!!")

try:
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()

    # data_transformation_df_config = DataTranformationConfig()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_path, test_path)


except Exception as e:
    logging.info("Custom Exception")
    raise CustomException(e, sys) # type: ignore