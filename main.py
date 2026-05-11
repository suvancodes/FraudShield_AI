from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig,DataValidationConfig
from src.components.data_validation import DataValidation
from src.exception.exciption import CustomException
import sys
from src.logger.logging import logging
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.artifact_entity import DataValidationArtifact
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact
from src.utils.ml_utils.text_preprocessor_utils import TextPreprocessorUtils

if __name__ == "__main__":
    try:
        data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion artifact: {data_ingestion_artifact}")
        print(data_ingestion_artifact)
        data_validation_config = DataValidationConfig()
        data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info(f"Data Validation artifact: {data_validation_artifact}")
        print(data_validation_artifact)
        data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation(data_transformation_config=data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation(data_ingestion_artifact=data_ingestion_artifact)
        logging.info(f"Data Transformation artifact: {data_transformation_artifact}")
        print(data_transformation_artifact)
    except Exception as e:
        raise CustomException(e, sys)







