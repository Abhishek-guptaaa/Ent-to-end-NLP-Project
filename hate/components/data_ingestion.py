import os
import sys
import pandas as pd
from hate.logger import logging
from hate.exception import CustomException
from dataclasses import dataclass
from mongo import read_mongo_data  # this is the mongo.py file

@dataclass
class DataIngestionConfig():
    train_data_path:str =os.path.join("artifacts", "train.csv")
    test_data_path:str =os.path.join("artifacts", "test.csv")
    raw_data_path:str =os.path.join("artifacts", "raw.csv")

class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # Reading Code
            logging.info("Reading from my mongo database")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) ## three data path choose any
        except Exception as e:
            raise CustomException(e , sys)
