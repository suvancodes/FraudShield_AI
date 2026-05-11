import numpy as np
import pandas as pd
import os
from src.constants import traning_pipeline
import sys
from src.logger.logging import logging
from src.exception.exciption import CustomException

class DataIngestionConfig:
    def __init__(self):
        try:
            logging.info("Data Ingestion configuartion started") 
            self.artifact_dir = traning_pipeline.ARTIFACT_DIR   
            self.data_ingestion_dir = os.path.join(self.artifact_dir, traning_pipeline.DATA_INGESTION_DIR_NAME)
            self.feature_store_dir = os.path.join(self.data_ingestion_dir, traning_pipeline.DATA_INGESTION_FEATURE_STORE_DIR)
            self.ingested_dir = os.path.join(self.data_ingestion_dir, traning_pipeline.DATA_INGESTION_INGESTED_DIR)
            self.train_test_split_ratio = traning_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
            
            self.raw_data_dir = os.path.join("data", "SMSSpamCollection")
            self.train_file_name = os.path.join(self.ingested_dir, traning_pipeline.TRANING_FILE_NAME)
            self.test_file_name = os.path.join(self.ingested_dir, traning_pipeline.TEST_FILE_NAME)
            self.file_name = os.path.join(self.feature_store_dir, traning_pipeline.FILE_NAME)
            
            logging.info("Data Ingestion configuartion completed")
        except Exception as e:
            raise CustomException(e, sys)
        
class DataValidationConfig:
    def __init__(self):
        try:
            logging.info("Data Validation configuartion started") 
            self.artifact_dir = traning_pipeline.ARTIFACT_DIR
            self.data_validation_dir = os.path.join(self.artifact_dir, traning_pipeline.DATA_VALIDATION_DIR_NAME)
            self.validated_dir = os.path.join(self.data_validation_dir, traning_pipeline.DATA_VALIDATION_VALID_DIR)
            self.invalid_dir = os.path.join(self.data_validation_dir, traning_pipeline.DATA_VALIDATION_INVALID_DIR)
            self.drift_report_dir = os.path.join(self.data_validation_dir, traning_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR)
            self.drift_report_file_name = os.path.join(self.drift_report_dir, traning_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
            
            logging.info("Data Validation configuartion completed")
        except Exception as e:
            raise CustomException(e, sys)
        
        
class DataTransformationConfig:
    def __init__(self):
        try:
            logging.info("Data Transformation configuartion started") 
            self.artifact_dir = traning_pipeline.ARTIFACT_DIR
            self.data_transformation_dir = os.path.join(self.artifact_dir, traning_pipeline.DATA_TRANSFORMATION_DIR_NAME)
            self.transformed_dir = os.path.join(self.data_transformation_dir, traning_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DIR)
            self.transformation_object_dir = os.path.join(self.data_transformation_dir, traning_pipeline.DATA_TRANSFORMATION_OBJECT_DIR)
            self.preprocessing_dir = os.path.join(self.data_transformation_dir, traning_pipeline.DATA_TRANSFORMATION_PREPROCESSING_DIR)
            self.preprocessing_object_file_name = os.path.join(self.preprocessing_dir, traning_pipeline.DATA_TRANSFORMATION_PREPROCESSING_OBJECT_FILE_NAME)
            self.transformed_data_dir = os.path.join(self.data_transformation_dir, traning_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DIR)
            self.transformed_train_file_name = os.path.join(self.transformed_data_dir, traning_pipeline.TRANING_FILE_NAME)
            self.transformed_test_file_name = os.path.join(self.transformed_data_dir, traning_pipeline.TEST_FILE_NAME)
            self.wordtovector_dir = os.path.join(self.data_transformation_dir, traning_pipeline.DATA_TRANSFORMATION_WORDTOVECTOR_DIR)
            self.wordtovector_object_file_name = os.path.join(self.wordtovector_dir, traning_pipeline.DATA_TRANSFORMATION_WORDTOVECTOR_OBJECT_FILE_NAME)
        except Exception as e:
            raise CustomException(e, sys)
        