import os
import pandas as pd
import numpy as np
import sys
from gensim.models import Word2Vec
from src.logger.logging import logging
from src.exception.exciption import CustomException
from src.entity.config_entity import DataTransformationConfig

class Word2VecUtils:
    def __init__(self, data_transformation_config: DataTransformationConfig):
        self.data_transformation_config = data_transformation_config

    def train_word2vec_model(self, sentences):
        """
        Trains a Word2Vec model.
        
        Args:
            sentences (pd.Series): A series of tokenized sentences (lists of words).
            
        Returns:
            A trained gensim Word2Vec model.
        """
        try:
            logging.info("Training Word2Vec model...")
            
            w2v_model = Word2Vec(
                sentences=sentences,
                vector_size=self.data_transformation_config.word2vec_vector_size,
                window=self.data_transformation_config.word2vec_window,
                min_count=self.data_transformation_config.word2vec_min_count
            )
            logging.info("Word2Vec model training completed.")
            return w2v_model
        except Exception as e:
            raise CustomException(e, sys)

    def vectorize_data(self, sentences, model):
        """
        Converts a series of tokenized sentences into averaged feature vectors.
        
        Args:
            sentences (pd.Series): A series of tokenized sentences.
            model: A trained gensim Word2Vec model.
            
        Returns:
            np.ndarray: An array of feature vectors.
        """
        try:
            logging.info("Vectorizing text data...")
            vectors = []
            for sentence in sentences:
                word_vectors = [model.wv[word] for word in sentence if word in model.wv]
                
                if not word_vectors:
                    vectors.append(np.zeros(model.vector_size))
                else:
                    vectors.append(np.mean(word_vectors, axis=0))
            
            logging.info("Text vectorization completed.")
            return np.array(vectors)
        except Exception as e:
            raise CustomException(e, sys)