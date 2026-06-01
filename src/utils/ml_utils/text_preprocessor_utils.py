import pandas as pd
import numpy as np
import os
import sys
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from src.constants import traning_pipeline
from src.logger.logging import logging
from src.exception.exciption import CustomException
from src.entity.config_entity import DataTransformationConfig

# Download stopwords if not already present
try:
    # This line will raise a LookupError if 'stopwords' is not found
    nltk.data.find('corpora/stopwords')
except LookupError:
    # FIX: Catch the correct exception (LookupError) and download the resource
    logging.info("Stopwords not found. Downloading...")
    nltk.download('stopwords')

class TextPreprocessorUtils:
    def __init__(self):
        """
        This class is now stateless and does not need the config.
        """
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text: str) -> list:
        """
        Cleans and tokenizes a single text string.
        
        Args:
            text (str): The input text message.
            
        Returns:
            list: A list of cleaned and stemmed words.
        """
        try:
            if not isinstance(text, str):
                text = str(text)
            
            text = re.sub('[^a-zA-Z]', ' ', text).lower()
            words = text.split()
            processed_words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
            
            return processed_words
            
        except Exception as e:
            logging.error(f"Error preprocessing text: '{text}'. Error: {e}")
            return []