# import os
# import sys
# import pandas as pd
# from hate.logger import logging
# from hate.exception import CustomException
# from dataclasses import dataclass
# from mongo import read_mongo_data  # this is the mongo.py file
# from sklearn.model_selection import train_test_split


# @dataclass
# class DataIngestionConfig():
#     train_data_path:str =os.path.join("artifacts", "train.csv")
#     test_data_path:str =os.path.join("artifacts", "test.csv")
#     raw_data_path:str =os.path.join("artifacts", "raw.csv")

# class DataIngestion():
#     def __init__(self):
#         self.ingestion_config = DataIngestionConfig()

#     def initiate_data_ingestion(self):
#         try:
#             df = read_mongo_data() # reading the data from mongo database
#             logging.info("Reading completed my mongo database")

#             os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) ## three data path choose any
#             df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

#             train_set, test_set=train_test_split(df, test_size=0.2, random_state=42)
#             df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
#             df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

#             logging.info("Data Ingestion is completed")

#             return(
#                 self.ingestion_config.train_data_path,
#                 self.ingestion_config.test_data_path
#             )
#         except Exception as e:
#             raise CustomException(e , sys)



import os
import sys
import pandas as pd
from hate.logger import logging
from hate.exception import CustomException
from dataclasses import dataclass
from mongo import read_mongo_data  # this is the mongo.py file
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig():
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")

class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df = read_mongo_data()  # reading the data from mongo database
            if df is None:
                logging.error("No data found to ingest")
                return None

            logging.info("Reading completed from mongo database")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

