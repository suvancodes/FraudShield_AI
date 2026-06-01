from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import os
import sys
from src.constants import traning_pipeline
from src.logger.logging import logging
from src.exception.exciption import CustomException
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.utils.ml_utils.text_preprocessor_utils import TextPreprocessorUtils
from src.utils.ml_utils.word2vec_utils import Word2VecUtils
from src.utils.main_utils.utils import save_object

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig):
        self.data_transformation_config = data_transformation_config
        
    def initiate_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation started")
            train_df = pd.read_csv(data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(data_ingestion_artifact.test_file_path)
            logging.info("Read train and test data completed.")

            # Initialize preprocessor and word2vec utilities
            # FIX: TextPreprocessorUtils no longer takes a config argument
            text_preprocessor_utils = TextPreprocessorUtils()
            word2vec_utils = Word2VecUtils(self.data_transformation_config)

            # Preprocess text
            train_df['processed_message'] = train_df['message'].apply(text_preprocessor_utils.preprocess_text)
            test_df['processed_message'] = test_df['message'].apply(text_preprocessor_utils.preprocess_text)
            logging.info("Text preprocessing completed.")

            # Train Word2Vec model ONLY on training data to prevent leakage
            logging.info("Training Word2Vec model on training data...")
            w2v_model = word2vec_utils.train_word2vec_model(train_df['processed_message'])
            logging.info("Word2Vec model training completed.")

            # Vectorize train and test data using the trained model
            X_train = word2vec_utils.vectorize_data(train_df['processed_message'], w2v_model)
            X_test = word2vec_utils.vectorize_data(test_df['processed_message'], w2v_model)
            y_train = train_df[traning_pipeline.TARGET_COLUMN].map({'ham': 0, 'spam': 1})
            y_test = test_df[traning_pipeline.TARGET_COLUMN].map({'ham': 0, 'spam': 1})
            logging.info("Text vectorization completed.")

            # Apply SMOTE to the training data
            logging.info(f"Before SMOTE, train distribution: {y_train.value_counts().to_dict()}")
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            logging.info(f"After SMOTE, train distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}")

            # Combine features and target into final arrays for training
            train_arr = np.c_[X_train_resampled, np.array(y_train_resampled)]
            test_arr = np.c_[X_test, np.array(y_test)]

            # Save the actual trained Word2Vec model
            os.makedirs(os.path.dirname(self.data_transformation_config.wordtovector_object_file_name), exist_ok=True)
            save_object(self.data_transformation_config.wordtovector_object_file_name, w2v_model)
            logging.info(f"Word2Vec model object saved at: {self.data_transformation_config.wordtovector_object_file_name}")
            
            # Save the text preprocessor utility (this one is stateless, so saving the class is okay)
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessing_object_file_name), exist_ok=True)
            save_object(self.data_transformation_config.preprocessing_object_file_name, text_preprocessor_utils)
            logging.info(f"Text preprocessor object saved at: {self.data_transformation_config.preprocessing_object_file_name}")

            # Save transformed data as numpy arrays
            os.makedirs(self.data_transformation_config.transformed_dir, exist_ok=True)
            np.save(self.data_transformation_config.transformed_train_file_name, train_arr)
            np.save(self.data_transformation_config.transformed_test_file_name, test_arr)
            logging.info("Saved transformed train and test arrays.")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_name,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_name,
                preprocessing_object_file_path=self.data_transformation_config.preprocessing_object_file_name,
                wordtovector_object_file_path=self.data_transformation_config.wordtovector_object_file_name
            )
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)