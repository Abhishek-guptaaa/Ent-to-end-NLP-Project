from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import joblib
import os
from config.config import Config
from hate.exception import CustomException
from hate.logger import logging

class TextTokenization:
    def __init__(self):
        self.tokenizer = Tokenizer(num_words=Config.MAX_WORDS)

    def fit_tokenizer(self, texts):
        try:
            logging.info("Fitting tokenizer on texts")
            self.tokenizer.fit_on_texts(texts)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(Config.TOKENIZER_PATH), exist_ok=True)
            
            joblib.dump(self.tokenizer, Config.TOKENIZER_PATH)
            logging.info(f"Tokenizer saved at {Config.TOKENIZER_PATH}")
        except Exception as e:
            raise CustomException("Error in fitting tokenizer", e)

    def transform_texts(self, texts):
        try:
            logging.info("Transforming texts to sequences")
            sequences = self.tokenizer.texts_to_sequences(texts)
            sequences_matrix = pad_sequences(sequences, maxlen=Config.MAX_LEN)
            return sequences_matrix
        except Exception as e:
            raise CustomException("Error in transforming texts", e)

    def load_tokenizer(self):
        try:
            self.tokenizer = joblib.load(Config.TOKENIZER_PATH)
        except Exception as e:
            raise CustomException("Error in loading tokenizer", e)
