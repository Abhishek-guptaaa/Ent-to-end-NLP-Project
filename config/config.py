# import os
# class Config:
#     RAW_DATA_PATH = 'notebook/raw.csv'
#     CLEANED_DATA_PATH = 'notebook/cleaned_data.csv'
#     TOKENIZER_PATH = 'models/tokenizer.pkl'
#     MODEL_PATH = 'models/nlp_model.h5'
#     LOG_PATH = 'logs/logs.log'
#     X_TEST_PATH = 'notebook/X_test.csv'
    
#     DROP_COLUMNS = ['id']
#     AXIS = 1
#     INPLACE = True
#     CLASS = 'class'
#     TWEET = 'tweet'
#     MAX_WORDS = 50000
#     MAX_LEN = 300

#     APP_HOST='0.0.0.0'
#     APP_PORT=8080


import os

class Config:
    RAW_DATA_PATH = 'notebook/raw.csv'
    CLEANED_DATA_PATH = 'notebook/cleaned_data.csv'
    TOKENIZER_PATH = 'models/tokenizer.pkl'
    MODEL_PATH = 'models/nlp_model.h5'
    LOG_PATH = 'logs/logs.log'
    X_TEST_PATH = 'notebook/X_test.csv'
    
    DROP_COLUMNS = ['id']
    AXIS = 1
    INPLACE = True
    CLASS = 'class'
    TWEET = 'tweet'
    MAX_WORDS = 50000
    MAX_LEN = 300

    APP_HOST = '0.0.0.0'
    APP_PORT = 8080
