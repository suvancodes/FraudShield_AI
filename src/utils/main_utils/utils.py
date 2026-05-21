import yaml
import sys
import os
import pickle  # or dill, if save_object uses dill
import numpy as np
import dill
from pathlib import Path
from src.entity.config_entity import DataIngestionConfig
from src.logger.logging import logging
from src.exception.exciption import CustomException


def save_object(file_path: str, obj: object) -> None:
    try:
        path = Path(file_path)

        # Ensure parent dirs exist (and fail fast if a file blocks the path)
        current = Path()
        for part in path.parent.parts:
            current = current / part
            if current.exists() and current.is_file():
                raise NotADirectoryError(f"Expected a directory but found a file at: {current}")

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)