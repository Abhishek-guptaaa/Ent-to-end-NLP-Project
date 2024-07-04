
# import pandas as pd
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
# from keras.utils import pad_sequences
# from keras.optimizers import RMSprop
# from config.config import Config
# from hate.logger import logging
# from hate.exception import CustomException
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import joblib

# class ModelTraining:
#     def __init__(self, sequences_matrix, labels):
#         self.sequences_matrix = sequences_matrix
#         self.labels = labels
#         self.model = None

#     def build_model(self):
#         try:
#             logging.info("Building LSTM model")
#             self.model = Sequential()
#             self.model.add(Embedding(Config.MAX_WORDS, 100, input_length=Config.MAX_LEN))
#             self.model.add(SpatialDropout1D(0.2))
#             self.model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
#             self.model.add(Dense(1, activation='sigmoid'))
#             self.model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
#             logging.info("Model built successfully")
#         except Exception as e:
#             logging.error(f"Error in building model: {e}")
#             raise CustomException("Error in building model", e)

#     def train_model(self):
#         try:
#             logging.info("Starting model training")

#             # Split the data into training and test sets first
#             X_train, X_test, y_train, y_test = train_test_split(self.sequences_matrix, self.labels, test_size=0.2, random_state=42)
#             logging.info(f"Data split into training and test sets: Train size: {len(X_train)}, Test size: {len(X_test)}")

#             # Now split the training data into training and validation sets
#             X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
#             logging.info(f"Training data split into training and validation sets: Train size: {len(X_train)}, Validation size: {len(X_val)}")

#             history = self.model.fit(X_train, y_train, batch_size=128, epochs=1, validation_data=(X_val, y_val))
#             self.model.save(Config.MODEL_PATH)
#             logging.info(f"Model training completed and saved at {Config.MODEL_PATH}")
#             return history, (X_test, y_test)
#         except Exception as e:
#             logging.error(f"Error in training model: {e}")
#             raise CustomException("Error in training model", e)

#     def preprocess_data(self, data):
#         # Implement preprocessing steps similar to training preprocessing
#         # This is a placeholder function and should be adjusted to match your actual preprocessing
#         return data

#     def evaluate_model(self):
#         try:
#             logging.info("Starting model evaluation")

#             # Load the test data from the CSV file
#             test_data = pd.read_csv(Config.CLEANED_DATA_PATH)
#             X_test = test_data[Config.TWEET]  # Adjust the column name
#             y_test = test_data[Config.CLASS]  # Adjust the column name

#             # Ensure the test data is preprocessed in the same way as training data
#             X_test = self.preprocess_data(X_test)  # Implement this method

#             # Load tokenizer and transform the test data
#             tokenizer = joblib.load(Config.TOKENIZER_PATH)
#             X_test = tokenizer.texts_to_sequences(X_test)
#             X_test = pad_sequences(X_test, maxlen=Config.MAX_LEN)

#             # Evaluate the model
#             y_pred = self.model.predict(X_test)
#             y_pred = (y_pred > 0.5).astype(int)

#             accuracy = accuracy_score(y_test, y_pred)
#             precision = precision_score(y_test, y_pred)
#             recall = recall_score(y_test, y_pred)
#             f1 = f1_score(y_test, y_pred)

#             logging.info(f"Model evaluation completed. Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

#             return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
#         except Exception as e:
#             logging.error(f"Error in evaluating model: {e}")
#             raise CustomException("Error in evaluating model", e)






import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from keras.utils import pad_sequences
from keras.optimizers import RMSprop
from config.config import Config
from hate.logger import logging
from hate.exception import CustomException
from sklearn.model_selection import train_test_split
import joblib

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

            # Split the data into training and test sets first
            X_train, X_test, y_train, y_test = train_test_split(self.sequences_matrix, self.labels, test_size=0.2, random_state=42)
            logging.info(f"Data split into training and test sets: Train size: {len(X_train)}, Test size: {len(X_test)}")

            # Now split the training data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            logging.info(f"Training data split into training and validation sets: Train size: {len(X_train)}, Validation size: {len(X_val)}")

            history = self.model.fit(X_train, y_train, batch_size=128, epochs=1, validation_data=(X_val, y_val))
            self.model.save(Config.MODEL_PATH)
            logging.info(f"Model training completed and saved at {Config.MODEL_PATH}")
            return history, (X_test, y_test)
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise CustomException("Error in training model", e)
