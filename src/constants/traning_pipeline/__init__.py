import os
import sys
import numpy as np
import pandas as pd

""" 
Data Ingestion releted consyant start with DATA_INJECTION VAR NAME
"""
RAW_DATA_DIR:str = "data/SMSSpamCollection" # Corrected path slightly for consistency
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2

"""
DEFINING COMMON CONSTANTS FOR TRAINING PIPELINE
"""

TARGET_COLUMN: str = "label"
PIPELINE_NAME: str = "src"
ARTIFACT_DIR: str = "artifacts"
FILE_NAME: str = "SMSSpamCollection"
TRAINING_FILE_NAME: str = "train.csv" # Added extension for clarity
TEST_FILE_NAME: str = "test.csv" # Added extension for clarity


# """
# Data validation releted constant 
# """
DATA_VALIDATION_DIR_NAME:str  = 'data_validation'
DATA_VALIDATION_VALID_DIR:str = 'validated'
DATA_VALIDATION_INVALID_DIR :str = 'invalid'
DATA_VALIDATION_DRIFT_REPORT_DIR:str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str = 'report.yaml'


# SCHEMA_FILE_PATH = os.path.join("data_schema","schema.yaml")



# """"Data transformation releted constant"""
DATA_TRANSFORMATION_DIR_NAME:str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR:str = "transformed"
DATA_TRANSFORMATION_PREPROCESSING_DIR:str = "preprocessing"
DATA_TRANSFORMATION_PREPROCESSING_OBJECT_FILE_NAME:str = "preprocessed_object.pkl"
DATA_TRANSFORMATION_WORDTOVECTOR_DIR:str = "wordtovector"
DATA_TRANSFORMATION_WORDTOVECTOR_OBJECT_FILE_NAME:str = "wordtovector_object.pkl"
TRANSFORMED_TRAIN_FILE_NAME: str = "train.npy" # Added for .npy files
TRANSFORMED_TEST_FILE_NAME: str = "test.npy" # Added for .npy files

# Word2Vec Hyperparameters
WORD2VEC_VECTOR_SIZE: int = 100
WORD2VEC_WINDOW: int = 5
WORD2VEC_MIN_COUNT: int = 2


# """model training releted constant"""
MODEL_TRAINER_DIR_NAME:str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR:str = "trained_model"
MODEL_TRAINER_EXPECTED_SCORE:float = 0.6
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD:float = 0.05
SAVE_MODEL_DIR:str = os.path.join("saved_models")
MODEL_FILE_NAME:str = "model.pkl"


