import re
import nltk
import string
import os 
import sys
import pandas as pd
from nltk.corpus import stopwords
nltk.download('stopwords')
from config.config import Config
from hate.exception import CustomException
from hate.logger import logging

class DataCleaning:
    def __init__(self):
        self.stemmer = nltk.SnowballStemmer("english")
        self.stopword = set(stopwords.words('english'))

    def clean_text(self, text):
        try:
            text = str(text).lower()
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            text = re.sub(r'<.*?>+', '', text)
            text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
            text = re.sub(r'\n', '', text)
            text = re.sub(r'\w*\d\w*', '', text)
            text = [word for word in text.split() if word not in self.stopword]
            text = " ".join(text)
            text = [self.stemmer.stem(word) for word in text.split()]
            text = " ".join(text)
            return text
        except Exception as e:
            raise CustomException("Error in data cleaning", e)

    def clean_data(self, df, text_column):
        try:
            if not isinstance(df, pd.DataFrame):
                raise CustomException("Input data is not a DataFrame", sys)
            
            if text_column not in df.columns:
                raise CustomException(f"'{text_column}' column not found in DataFrame", sys)
            
            df[text_column] = df[text_column].apply(self.clean_text)
            return df
        except Exception as e:
            logging.info(f"Error in cleaning data frame: {e}")
            raise CustomException("Error in cleaning data frame", e)
