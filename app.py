from src.ML_Project.logger import logging
from src.ML_Project.exception import CustomException
import sys
from src.ML_Project.components.Data_Ingestion import DataIngestion
from src.ML_Project.components.Data_Tranformation import DataTranformationConfig,DataTransformation
from src.ML_Project.components.Model_Trainer import ModelTrainerConfig, ModelTrainer


if __name__ == "__main__":
    logging.info("The execution has started!!")

try:
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()

    # data_transformation_df_config = DataTranformationConfig()
    data_transformation = DataTransformation()
    train_arr, test_arr,_=data_transformation.initiate_data_transformation(train_path, test_path)


    ## Model Training

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))


except Exception as e:
    logging.info("Custom Exception")
    raise CustomException(e, sys) # type: ignore


