import pandas as pd
import numpy as np
import os
import sys
from src.constants import traning_pipeline
from src.entity.config_entity import DataIngestionConfig
from src.logger.logging import logging
from src.exception.exciption import CustomException
from src.entity.artifact_entity import DataIngestionArtifact

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def initiate_data_ingestion(self):
        try:
            logging.info("Data Ingestion started")
            df = pd.read_csv(
                self.data_ingestion_config.raw_data_dir,
                sep="\t",
                header=None,
                names=["label", "message"],
            )
            logging.info("Data read from source completed")

            # create feature store dir
            os.makedirs(self.data_ingestion_config.feature_store_dir, exist_ok=True)

            # save data in feature store dir
            df.to_csv(self.data_ingestion_config.file_name, index=False)
            logging.info("Data saved in feature store completed")

            # split data into train and test
            from sklearn.model_selection import train_test_split

            train_df, test_df = train_test_split(df, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=42)

            # create ingested dir
            os.makedirs(self.data_ingestion_config.ingested_dir, exist_ok=True)

            # save train and test data in ingested dir
            train_df.to_csv(self.data_ingestion_config.train_file_name, index=False)
            test_df.to_csv(self.data_ingestion_config.test_file_name, index=False)
            logging.info("Data split into train and test completed")

            # create data ingestion artifact
            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.file_name,
                train_file_path=self.data_ingestion_config.train_file_name,
                test_file_path=self.data_ingestion_config.test_file_name
            )

            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys)