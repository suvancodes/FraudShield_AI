import os
import sys

# Force project root to be FIRST on sys.path (prevents importing another installed `src`)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

from src.exception.exciption import CustomException
from src.logger.logging import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)


class TrainPipeline:
    def start_training_pipeline(self):
        try:
            # Data Ingestion
            data_ingestion = DataIngestion(data_ingestion_config=DataIngestionConfig())
            ingestion_artifact = data_ingestion.initiate_data_ingestion()

            # Data Validation
            data_validation = DataValidation(
                data_validation_config=DataValidationConfig(),
                data_ingestion_artifact=ingestion_artifact,
            )
            _ = data_validation.initiate_data_validation()

            # Data Transformation
            data_transformation_config = DataTransformationConfig()
            data_transformation = DataTransformation(data_transformation_config)
            transformation_artifact = data_transformation.initiate_data_transformation(
                data_ingestion_artifact=ingestion_artifact
            )

            # Model Training
            model_trainer_config = ModelTrainerConfig()
            model_trainer = ModelTrainer(
                data_transformation_config=data_transformation_config,
                model_trainer_config=model_trainer_config,
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer(
                transformation_artifact
            )

            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    logging.info("Starting training pipeline...")
    TrainPipeline().start_training_pipeline()