import pandas as pd
import numpy as np
import os
import sys
from src.constants import traning_pipeline
from src.entity.config_entity import DataIngestionConfig
from src.logger.logging import logging
from src.exception.exciption import CustomException
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.artifact_entity import DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.utils.ml_utils.text_preprocessor_utils import TextPreprocessorUtils
from src.utils.ml_utils.word2vec_utils import Word2VecUtils

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig):
        self.data_transformation_config = data_transformation_config
    def initiate_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation started")
            # read train and test data
            train_df = pd.read_csv(data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(data_ingestion_artifact.test_file_path)

            all_df = pd.concat([train_df, test_df], axis=0)
            
            text_preprocessor_utils = TextPreprocessorUtils(self.data_transformation_config)
            train_df, test_df, all_df = text_preprocessor_utils.preprocess_text(train_df, test_df, all_df)
            
            word2vec_utils = Word2VecUtils(self.data_transformation_config)
            train_df, test_df = word2vec_utils.word2vec(train_df, test_df, all_df)

            # create transformed dir
            os.makedirs(self.data_transformation_config.transformed_dir, exist_ok=True)

            # save transformed train and test data
            train_df.to_csv(self.data_transformation_config.transformed_train_file_name, index=False)
            test_df.to_csv(self.data_transformation_config.transformed_test_file_name, index=False)
            logging.info("Data Transformation completed")
            # create data transformation artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_name,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_name,
                preprocessing_object_file_path=self.data_transformation_config.preprocessing_object_file_name,
                wordtovector_object_file_path=self.data_transformation_config.wordtovector_object_file_name
            )
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)