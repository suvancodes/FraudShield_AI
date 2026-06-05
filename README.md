# FraudShield AI: End-to-End SMS Spam Detection

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-FF4B4B.svg?style=for-the-badge&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E.svg?style=for-the-badge&logo=scikit-learn)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

An end-to-end machine learning project that builds and deploys a robust system to classify SMS messages as "Spam" or "Ham" (not spam). The project features a complete MLOps pipeline, from data ingestion and validation to model training and deployment as an interactive web application.

[**Live Demo**](https://fraudshield-suvankar.streamlit.app/)

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run Locally](#how-to-run-locally)
- [The Pipeline Explained](#the-pipeline-explained)

## Overview

This project tackles the common problem of SMS spam by implementing a full machine learning pipeline. It processes raw text data, transforms it into meaningful numerical representations using Word2Vec, trains multiple classification models to find the best performer, and serves the final model through a user-friendly Streamlit web interface for real-time predictions.

## Key Features

- **End-to-End ML Pipeline**: A modular pipeline covering Data Ingestion, Data Validation, Data Transformation, Model Training, and Model Evaluation.
- **Advanced NLP Preprocessing**: Utilizes NLTK for text cleaning, tokenization, stop-word removal, and stemming to prepare text for modeling.
- **Word Embeddings with Word2Vec**: Employs `gensim`'s Word2Vec to capture semantic relationships between words, creating rich feature vectors for each message.
- **Handles Class Imbalance**: Implements SMOTE (Synthetic Minority Over-sampling Technique) to address the imbalanced nature of the spam dataset, preventing model bias towards the majority class.
- **Automated Model Selection**: Trains and evaluates several classifiers (e.g., Logistic Regression, Random Forest, Gradient Boosting) and automatically selects the best-performing model based on F1-score.
- **Interactive Web Application**: A web app built with Streamlit provides a clean UI for both single-message classification and batch prediction via CSV upload.

## Tech Stack

- **Language**: Python 3.11
- **Machine Learning**: Scikit-learn, Imbalanced-learn, XGBoost
- **NLP**: NLTK, Gensim
- **Web Framework**: Streamlit
- **Data Handling**: Pandas, NumPy
- **Utilities**: Dill, PyYAML, TQDM

## Project Structure

The project follows a modular structure to ensure scalability and maintainability.

```
├── artifacts/              # Stores output files from the pipeline (models, datasets)
├── data/                   # Raw and processed data
├── src/
│   ├── components/         # Core pipeline components (ingestion, training, etc.)
│   ├── constants/          # Project constants and configurations
│   ├── entity/             # Data classes for configuration and artifacts
│   ├── exception/          # Custom exception handling
│   ├── logger/             # Custom logging setup
│   └── utils/              # Utility functions
├── .gitignore              # Files to be ignored by Git
├── app.py                  # The Streamlit web application
├── main.py                 # Main script to run the training pipeline
├── requirements.txt        # Project dependencies
└── README.md               # You are here!
```

## How to Run Locally

Follow these steps to set up and run the project on your local machine.

**1. Clone the Repository**
```bash
git clone https://github.com/suvancodes/FraudShield_AI.git
cd FraudShield_AI
```

**2. Create and Activate a Virtual Environment**
```bash
# For conda
conda create -n spamenv python=3.11 -y
conda activate spamenv

# For venv
python -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the Training Pipeline**
This command will execute the entire ML pipeline, from data ingestion to model training, and save the final artifacts in the `artifacts/` directory.
```bash
python main.py
```

**5. Launch the Streamlit Application**
Once the pipeline has run successfully, launch the web app.
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

## The Pipeline Explained

1.  **Data Ingestion**: Fetches the raw `SMSSpamCollection` dataset, splits it into training and testing sets, and saves them as CSV files.
2.  **Data Validation**: Checks for data drift between the training and testing sets and validates the data schema.
3.  **Data Transformation**:
    -   Applies text preprocessing (lowercase, remove punctuation, tokenize, remove stopwords, stem).
    -   Trains a Word2Vec model on the processed text corpus to learn word embeddings.
    -   Converts each message into a numerical vector by averaging the vectors of its words.
    -   Saves the preprocessing object and the trained Word2Vec model as `.pkl` files.
4.  **Model Training**:
    -   Applies SMOTE to the training data to handle class imbalance.
    -   Trains multiple classification models on the resampled data.
    -   Evaluates each model using the F1-score and identifies the best performer.
    -   Saves the best-performing model as `model.pkl`.
<<<<<<< HEAD
5.  **Prediction**: The Streamlit app loads the saved preprocessing object, Word2Vec model, and classifier model to make predictions on new, unseen user input.
=======
5.  **Prediction**: The Streamlit app loads the saved preprocessing object, Word2Vec model, and classifier model to make predictions on new, unseen user input.
>>>>>>> 637890e9d5b0266b804c74bd61f32768a5e55db2
