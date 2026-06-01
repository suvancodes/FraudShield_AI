import pandas as pd
import numpy as np
import os
import sys
from src.constants import traning_pipeline
from src.entity.config_entity import DataIngestionConfig,ModelTrainerConfig
from src.logger.logging import logging
from src.exception.exciption import CustomException
from src.entity.artifact_entity import DataIngestionArtifact, ModelTrainerArtifact, DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.utils.main_utils.utils import save_object

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score 



class ModelTrainer:
    def __init__(self, data_transformation_config: DataTransformationConfig,model_trainer_config: ModelTrainerConfig):
        self.data_transformation_config = data_transformation_config
        self.model_trainer_config = model_trainer_config
        
    def initiate_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logging.info("Model Trainer started")
            logging.info("Reading transformed train and test data")
            
            # Load numpy arrays
            train_arr = np.load(data_transformation_artifact.transformed_train_file_path)
            test_arr = np.load(data_transformation_artifact.transformed_test_file_path)

            # split features and target variable from train and test data
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            
            logging.info("Data loaded and split successfully.")
            
            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "K-Nearest Neighbors": KNeighborsClassifier()
            }
            best_model_name = None
            best_model = None
            best_model_score = 0.0

            
            for i in range(len(models)):
                model_name = list(models.keys())[i]
                model = list(models.values())[i]
                logging.info(f"Training {model_name} started")
                model.fit(X_train, y_train)
                logging.info(f"Training {model_name} completed")
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                train_f1_score = f1_score(y_train, y_pred_train)
                train_precision_score = precision_score(y_train, y_pred_train)
                train_recall_score = recall_score(y_train, y_pred_train)
                test_f1_score = f1_score(y_test, y_pred_test)
                test_precision_score = precision_score(y_test, y_pred_test)
                test_recall_score = recall_score(y_test, y_pred_test)
                logging.info(f"{model_name} train f1 score: {train_f1_score}")
                logging.info(f"{model_name} train precision score: {train_precision_score}")
                logging.info(f"{model_name} train recall score: {train_recall_score}")
                logging.info(f"{model_name} test f1 score: {test_f1_score}")
                logging.info(f"{model_name} test precision score: {test_precision_score}")
                logging.info(f"{model_name} test recall score: {test_recall_score}")
                if test_f1_score > best_model_score:
                    best_model_name = model_name
                    best_model_score = test_f1_score
                    best_model = model
                    
            print(f"Best model: {best_model_name} with f1 score: {best_model_score}")
            logging.info(f"Best model found: {best_model_name} with f1 score: {best_model_score}")
            
            if best_model_score < self.model_trainer_config.expected_score:
                raise Exception("No best model found with expected accuracy")

            logging.info("Saving the best model")
            # FIX: Use the correct attribute name 'trained_model_file_path'
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            # create model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                best_model_name=best_model_name,
                best_model_score=best_model_score
            )
            
            return model_trainer_artifact
            
        except Exception as e:
            raise CustomException(e, sys)