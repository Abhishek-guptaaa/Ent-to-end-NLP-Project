import os
class Config:
    RAW_DATA_PATH = 'notebook/raw.csv'
    CLEANED_DATA_PATH = 'notebook/cleaned_data.csv'
    TOKENIZER_PATH = 'models/tokenizer.pkl'
    MODEL_PATH = 'models/nlp_model.h5'
    LOG_PATH = 'logs/logs.log'

    DROP_COLUMNS = ['id']
    AXIS = 1
    INPLACE = True
    CLASS = 'class'
    TWEET = 'tweet'
    DATA_TRANSFORMATION_ARTIFACTS_DIR = 'data_transformation_artifacts'
    TRANSFORMED_FILE_PATH = 'data_transformation_artifacts/transformed_data.csv'

    MAX_WORDS = 50000
    MAX_LEN = 300

