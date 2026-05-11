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

class TextPreprocessorUtils:
    def __init__(self, data_transformation_config: DataTransformationConfig):
        self.data_transformation_config = data_transformation_config
    def preprocess_text(self, train_df: pd.DataFrame, test_df: pd.DataFrame, all_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            logging.info("Text Preprocessing started")
            # text preprocessing
            import re
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import PorterStemmer

            nltk.download("stopwords")
            stop_words = set(stopwords.words("english"))
            ps = PorterStemmer()

            def preprocess_text(text):
                text = re.sub(r"[^a-zA-Z]", " ", text)
                text = text.lower()
                text = text.split()
                text = [ps.stem(word) for word in text if word not in stop_words]
                text = " ".join(text)
                return text

            train_word = train_df['message'].apply(preprocess_text)
            test_word = test_df['message'].apply(preprocess_text)
            all_word = all_df['message'].apply(preprocess_text)
            
            train_df['message'] = train_word
            test_df['message'] = test_word
            all_df['message'] = all_word
            logging.info("Text Preprocessing completed")
            return train_df, test_df, all_df
        except Exception as e:
            raise CustomException(e, sys)