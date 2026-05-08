import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str
    train_file_path: str
    test_file_path: str
    
    
@dataclass
class DataValidationArtifact:
    validated_train_file_path: str
    validated_test_file_path: str
    drift_report_file_path: str
    
@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str
    preprocessing_object_file_path: str