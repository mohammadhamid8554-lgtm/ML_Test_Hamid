import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler  
from sklearn.impute import SimpleImputer  
from sklearn.pipeline import Pipeline  
from src.ML_Project.exception import CustomException
from src.ML_Project.logger import logging
from sklearn.compose import ColumnTransformer  
from src.ML_Project.utils import save_obj
import os




@dataclass
class DataTranformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "Preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTranformationConfig()

    def get_data_tranformer_object(self):
        """
        this function is responsible for data tranformation
        """
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = ["gender", "race/ethnicity", "parental level of education",
                                   "lunch", "test preparation course"]

            num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                           ("scaler", StandardScaler())])

            cat_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                           ("one_hot_encoder", OneHotEncoder()),
                                           ("scaler", StandardScaler(with_mean=False))])

            logging.info(f"Categorical Columns:{categorical_columns}")
            logging.info(f"Numerical Columns: {numerical_columns}")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)  # type: ignore

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train and test file")

            preprocessing_obj = self.get_data_tranformer_object()

            target_column_name = "math score"


           # divide the train data to independent and dependent feature
           
            input_features_train_df = train_df.drop(columns = [target_column_name])
            target_feature_train_df = train_df[target_column_name]

           # divide the test data to independent and dependent feature 

            input_features_test_df = test_df.drop(columns = [target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying Preprocessing on training and test dataset")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object")

            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys) # type: ignore

    initiate_data_tranformation = initiate_data_transformation



        
