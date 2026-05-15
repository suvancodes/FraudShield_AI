import yaml
import sys
import os
import pickle  # or dill, if save_object uses dill
import numpy as np
import dill
from src.entity.config_entity import DataIngestionConfig
from src.logger.logging import logging
from src.exception.exciption import CustomException


def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info(f"Saving object to file: {file_path}")
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Object saved successfully")
    except Exception as e:
        logging.error(f"Error saving object to file: {e}")
        raise CustomException(e, sys)