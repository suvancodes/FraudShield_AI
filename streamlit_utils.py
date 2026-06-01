import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys

# Add src to path to import custom classes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from src.utils.ml_utils.text_preprocessor_utils import TextPreprocessorUtils
from src.entity.config_entity import DataTransformationConfig, ModelTrainerConfig # Import config classes

@st.cache_resource
def load_artifacts():
    """
    Loads the necessary artifacts for prediction by using the pipeline's own config classes.
    """
    try:
        # Use the config classes to get the correct, dynamic paths
        data_transformation_config = DataTransformationConfig()
        model_trainer_config = ModelTrainerConfig()

        preprocessor_path = data_transformation_config.preprocessing_object_file_name
        w2v_model_path = data_transformation_config.wordtovector_object_file_name
        model_path = model_trainer_config.trained_model_file_path # Use the config path

        st.info(f"Loading preprocessor from: {preprocessor_path}")
        st.info(f"Loading Word2Vec model from: {w2v_model_path}")
        st.info(f"Loading classifier from: {model_path}")

        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)
            
        with open(w2v_model_path, "rb") as f:
            w2v_model = pickle.load(f)

        with open(model_path, "rb") as f:
            model = pickle.load(f)
            
        return preprocessor, w2v_model, model
    except FileNotFoundError as e:
        st.error(f"Artifact not found: {e}. Please ensure the training pipeline has been run successfully.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None

def predict_one(text: str, preprocessor, w2v_model, model, threshold: float = 0.5):
    """
    Makes a prediction for a single text input.
    """
    if not all([preprocessor, w2v_model, model]):
        return "Error: Models not loaded.", 0.0

    try:
        # 1. Preprocess the text using the loaded preprocessor
        processed_text = preprocessor.preprocess_text(text)
        
        # 2. Vectorize the processed text using the loaded Word2Vec model
        word_vectors = [w2v_model.wv[word] for word in processed_text if word in w2v_model.wv]
        
        if not word_vectors:
            # If no words are in the vocabulary, create a zero vector
            vector = np.zeros(w2v_model.vector_size).reshape(1, -1)
        else:
            # Average the word vectors to get a sentence vector
            vector = np.mean(word_vectors, axis=0).reshape(1, -1)
            
        # 3. Predict using the final classifier model
        probability = model.predict_proba(vector)[0][1] # Probability of being spam (class 1)
        prediction = "Spam" if probability >= threshold else "Ham"
        
        return prediction, probability
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return "Error", 0.0


