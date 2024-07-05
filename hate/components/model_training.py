


import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from config.config import Config
from hate.logger import logging
from hate.exception import CustomException

class ModelTraining:
    def __init__(self, sequences_matrix, labels):
        self.sequences_matrix = sequences_matrix
        self.labels = labels
        self.model = None

    def build_model(self):
        try:
            logging.info("Building LSTM model")
            self.model = Sequential()
            self.model.add(Embedding(Config.MAX_WORDS, 100, input_length=Config.MAX_LEN))
            self.model.add(SpatialDropout1D(0.2))
            self.model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
            self.model.add(Dense(1, activation='sigmoid'))
            self.model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
            logging.info("Model built successfully")
        except Exception as e:
            logging.error(f"Error in building model: {e}")
            raise CustomException("Error in building model", e)

    def train_model(self):
        try:
            logging.info("Starting model training")

            # Split the data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(self.sequences_matrix, self.labels, test_size=0.2, random_state=42)
            logging.info(f"Data split into training and validation sets: Train size: {len(X_train)}, Validation size: {len(X_val)}")

            history = self.model.fit(X_train, y_train, batch_size=128, epochs=1, validation_data=(X_val, y_val))
            self.model.save(Config.MODEL_PATH)
            logging.info(f"Model training completed and saved at {Config.MODEL_PATH}")

            return history
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise CustomException("Error in training model", e)





