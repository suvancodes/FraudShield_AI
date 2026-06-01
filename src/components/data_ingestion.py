import pandas as pd
import numpy as np
import os
import sys
import csv
from src.constants import traning_pipeline
from src.entity.config_entity import DataIngestionConfig
from src.logger.logging import logging
from src.exception.exciption import CustomException
from src.entity.artifact_entity import DataIngestionArtifact

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            raw_path = self.data_ingestion_config.raw_data_dir

            # Robust reader for SMSSpamCollection variants (tab-separated, quoted CSV, or malformed header)
            rows = []
            with open(raw_path, "r", encoding="latin-1", newline="") as f:
                reader = csv.reader(f, quotechar='"', skipinitialspace=True)
                for row in reader:
                    if not row:
                        continue
                    # If entire file had a header line like label,message skip it
                    first = row[0].strip().lower()
                    if first == "label" and len(row) >= 2 and row[1].strip().lower() == "message":
                        continue
                    # If csv.reader returned at least two fields, use them
                    if len(row) >= 2:
                        label = row[0]
                        msg = ",".join(row[1:])  # preserve commas inside message
                        rows.append([label, msg])
                        continue
                    # single-field row: try to recover by splitting on tab or first comma
                    field = row[0].strip()
                    # handle quoted single-field like: "ham,Message text"
                    if field.startswith('"') and field.endswith('"'):
                        field = field[1:-1]
                    if "\t" in field:
                        parts = field.split("\t", 1)
                    else:
                        parts = field.split(",", 1)
                    if len(parts) == 2:
                        rows.append([parts[0], parts[1]])
                        continue
                    # otherwise skip unparsable line
                    continue

            df = pd.DataFrame(rows, columns=["label", "message"])

            # Normalize types, strip quotes and whitespace
            df["label"] = df["label"].astype(str).str.strip().str.strip('"').str.lower()
            df["message"] = df["message"].fillna("").astype(str).str.strip().str.strip('"')

            # create feature store dir
            os.makedirs(self.data_ingestion_config.feature_store_dir, exist_ok=True)

            # save data in feature store dir
            df.to_csv(self.data_ingestion_config.file_name, index=False)
            logging.info("Data saved in feature store completed")

            # split data into train and test
            from sklearn.model_selection import train_test_split

            train_df, test_df = train_test_split(
                df, 
                test_size=self.data_ingestion_config.train_test_split_ratio, 
                random_state=42,
                stratify=df["label"]  # Add this line
            )

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