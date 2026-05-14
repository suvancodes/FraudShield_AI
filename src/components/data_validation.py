import pandas as pd
import numpy as np
import os
import sys
from src.constants import traning_pipeline
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
from src.logger.logging import logging
from src.exception.exciption import CustomException
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact



class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        self.data_validation_config = data_validation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Data Validation started")
            # read train and test data
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info("Data read from source completed")

            # create validated dir
            os.makedirs(self.data_validation_config.validated_dir, exist_ok=True)

            # save validated train and test data in validated dir
            validated_train_file_path = os.path.join(self.data_validation_config.validated_dir, traning_pipeline.TRAINING_FILE_NAME)
            validated_test_file_path = os.path.join(self.data_validation_config.validated_dir, traning_pipeline.TEST_FILE_NAME)
            train_df.to_csv(validated_train_file_path, index=False)
            test_df.to_csv(validated_test_file_path, index=False)
            logging.info("Data saved in validated dir completed")

            # create drift report dir
            os.makedirs(self.data_validation_config.drift_report_dir, exist_ok=True)

            # save drift report in drift report dir
            drift_report_file_path = os.path.join(self.data_validation_config.drift_report_dir, traning_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
            with open(drift_report_file_path, "w") as f:
                f.write("Drift report")

            logging.info("Drift report saved in drift report dir completed")

            # create data validation artifact
            data_validation_artifact = DataValidationArtifact(
                validated_train_file_path=validated_train_file_path,
                validated_test_file_path=validated_test_file_path,
                drift_report_file_path=drift_report_file_path
            )

            return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys)