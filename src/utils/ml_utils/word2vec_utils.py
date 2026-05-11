import pandas as pd
import numpy as np
import gensim
from typing import Tuple
import sys

from src.logger.logging import logging
from src.exception.exciption import CustomException
from src.entity.config_entity import DataTransformationConfig


class Word2VecUtils:

    def __init__(self, data_transformation_config: DataTransformationConfig):
        self.data_transformation_config = data_transformation_config

    def word2vec(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        all_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        try:
            logging.info("Word2Vec started")

            # Train model
            model = gensim.models.Word2Vec(
                sentences=all_df['message'],
                vector_size=100,
                window=5,
                min_count=1,
                workers=4
            )

            def avg_word2vec(doc):

                vectors = [
                    model.wv[word]
                    for word in doc
                    if word in model.wv
                ]

                if len(vectors) == 0:
                    return np.zeros(model.vector_size)

                return np.mean(vectors, axis=0)

            # Convert text into vectors
            train_vectors = train_df["message"].apply(avg_word2vec)
            test_vectors = test_df["message"].apply(avg_word2vec)

            # Convert vectors into dataframe
            train_vector_df = pd.DataFrame(train_vectors.tolist())
            test_vector_df = pd.DataFrame(test_vectors.tolist())
            
            train_vector_df["label"] = train_df["label"].apply(lambda x: 1 if x == "spam" else 0)
            test_vector_df["label"] = test_df["label"].apply(lambda x: 1 if x == "spam" else 0)

            logging.info("Word2Vec completed")

            return train_vector_df, test_vector_df

        except Exception as e:
            raise CustomException(e, sys)