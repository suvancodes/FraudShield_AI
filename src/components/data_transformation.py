from imblearn.over_sampling import SMOTE
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
        
    def apply_smote(self, input_df: pd.DataFrame, target_series: pd.Series) -> tuple:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(input_df, target_series)
        print("After SMOTE, the shape of X: ", X_resampled.shape)
        print("After SMOTE, the shape of y: ", y_resampled.shape)
        print("After SMOTE, the distribution of target variable: ", np.bincount(y_resampled))
        logging.info("SMOTE applied successfully")
        logging.info(f"After SMOTE, the shape of X: {X_resampled.shape}")
        logging.info(f"After SMOTE, the shape of y: {y_resampled.shape}")
        logging.info(f"After SMOTE, the distribution of target variable: {np.bincount(y_resampled)}")
        return X_resampled, y_resampled

    def initiate_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation started")
            logging.info("Reading train and test data")
            # read train and test data
            train_df = pd.read_csv(data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(data_ingestion_artifact.test_file_path)

            all_df = pd.concat([train_df, test_df], axis=0)
            logging.info("Data read from source completed")
            
            logging.info("Preprocessing text data...")
            
            text_preprocessor_utils = TextPreprocessorUtils(self.data_transformation_config)
            train_df, test_df, all_df = text_preprocessor_utils.preprocess_text(train_df, test_df, all_df)
            logging.info("Text preprocessing completed")
            
            logging.info("Applying word2vec...")
            word2vec_utils = Word2VecUtils(self.data_transformation_config)
            train_df, test_df = word2vec_utils.word2vec(train_df, test_df, all_df)
            logging.info("Word2Vec transformation completed")
            
            logging.info("Applying SMOTE to handle class imbalance...")
            logging.info(f"Before SMOTE, the shape of train data: {train_df.shape}")
            logging.info(f"Before SMOTE, the distribution of target variable: {train_df[traning_pipeline.TARGET_COLUMN].value_counts()}")
            
            
            # apply smote on train data
            X_train = train_df.drop(columns=[traning_pipeline.TARGET_COLUMN])
            y_train = train_df[traning_pipeline.TARGET_COLUMN]
            X_train_resampled, y_train_resampled = self.apply_smote(X_train, y_train)
            train_df = pd.concat([pd.DataFrame(X_train_resampled, columns=X_train.columns), pd.DataFrame(y_train_resampled, columns=[traning_pipeline.TARGET_COLUMN])], axis=1)
            logging.info(f"After SMOTE, the shape of train data: {train_df.shape}")
            logging.info(f"After SMOTE, the distribution of target variable: {train_df[traning_pipeline.TARGET_COLUMN].value_counts()}")
            
            # apply smote on test data
            logging.info(f"Before SMOTE, the shape of test data: {test_df.shape}")
            logging.info(f"Before SMOTE, the distribution of target variable: {test_df[traning_pipeline.TARGET_COLUMN].value_counts()}")
            X_test = test_df.drop(columns=[traning_pipeline.TARGET_COLUMN])
            y_test = test_df[traning_pipeline.TARGET_COLUMN]
            X_test_resampled, y_test_resampled = self.apply_smote(X_test, y_test)
            test_df = pd.concat([pd.DataFrame(X_test_resampled, columns=X_test.columns), pd.DataFrame(y_test_resampled, columns=[traning_pipeline.TARGET_COLUMN])], axis=1)
            logging.info(f"After SMOTE, the shape of test data: {test_df.shape}")
            logging.info(f"After SMOTE, the distribution of target variable: {test_df[traning_pipeline.TARGET_COLUMN].value_counts()}")

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