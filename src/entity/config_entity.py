import numpy as np
import pandas as pd
import os
from dataclasses import dataclass, field
from src.constants import traning_pipeline
import sys
from src.logger.logging import logging
from src.exception.exciption import CustomException

@dataclass
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
            self.train_file_name = os.path.join(self.ingested_dir, traning_pipeline.TRAINING_FILE_NAME)
            self.test_file_name = os.path.join(self.ingested_dir, traning_pipeline.TEST_FILE_NAME)
            self.file_name = os.path.join(self.feature_store_dir, traning_pipeline.FILE_NAME)
            
            logging.info("Data Ingestion configuartion completed")
        except Exception as e:
            raise CustomException(e, sys)
        
@dataclass
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
        
        
@dataclass
class DataTransformationConfig:
    # This class now directly defines its paths using constants
    # It no longer needs a custom __init__ method
    
    transformed_dir: str = os.path.join(traning_pipeline.ARTIFACT_DIR, traning_pipeline.DATA_TRANSFORMATION_DIR_NAME, traning_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DIR)
    
    transformed_train_file_name: str = os.path.join(transformed_dir, traning_pipeline.TRANSFORMED_TRAIN_FILE_NAME)
    
    transformed_test_file_name: str = os.path.join(transformed_dir, traning_pipeline.TRANSFORMED_TEST_FILE_NAME)
    
    preprocessing_object_file_name: str = os.path.join(
        traning_pipeline.ARTIFACT_DIR, 
        traning_pipeline.DATA_TRANSFORMATION_DIR_NAME, 
        traning_pipeline.DATA_TRANSFORMATION_PREPROCESSING_DIR, 
        traning_pipeline.DATA_TRANSFORMATION_PREPROCESSING_OBJECT_FILE_NAME
    )
    
    wordtovector_object_file_name: str = os.path.join(
        traning_pipeline.ARTIFACT_DIR, 
        traning_pipeline.DATA_TRANSFORMATION_DIR_NAME, 
        traning_pipeline.DATA_TRANSFORMATION_WORDTOVECTOR_DIR, 
        traning_pipeline.DATA_TRANSFORMATION_WORDTOVECTOR_OBJECT_FILE_NAME
    )
    
    # Add the Word2Vec hyperparameters
    word2vec_vector_size: int = traning_pipeline.WORD2VEC_VECTOR_SIZE
    word2vec_window: int = traning_pipeline.WORD2VEC_WINDOW
    word2vec_min_count: int = traning_pipeline.WORD2VEC_MIN_COUNT


@dataclass
class ModelTrainerConfig:
    # This class now directly defines its paths using constants
    
    trained_model_dir: str = os.path.join(
        traning_pipeline.ARTIFACT_DIR, 
        traning_pipeline.MODEL_TRAINER_DIR_NAME, 
        traning_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR
    )

    trained_model_file_path: str = os.path.join(
        trained_model_dir, 
        traning_pipeline.MODEL_FILE_NAME
    )

    expected_score: float = traning_pipeline.MODEL_TRAINER_EXPECTED_SCORE
    
    overfitting_underfitting_threshold: float = traning_pipeline.MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD


@dataclass
class ModelEvaluationConfig:
    # ... (assuming this class exists or will be added later)
    pass

@dataclass
class ModelPusherConfig:
    # ... (assuming this class exists or will be added later)
    pass