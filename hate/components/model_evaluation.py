import pandas as pd
import numpy as np
import os
import sys
import pickle
from hate.logger import logging
from hate.exception import CustomException
from hate.components.data_cleaning import DataCleaning
from hate.components.text_tokenization import TextTokenization
from config.config import Config
from keras.models import load_model
from keras.utils import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from hate.components.model_training import ModelTraining  # Corrected import statement

# Model Evaluation class
class ModelEvaluation:
    def __init__(self, config):
        self.config = config

    def load_model(self):
        try:
            model = load_model(self.config.MODEL_PATH)
            return model
        except Exception as e:
            raise CustomException(f"Failed to load model: {e}", sys)

    def load_tokenizer(self):
        try:
            with open(self.config.TOKENIZER_PATH, 'rb') as file:
                tokenizer = pickle.load(file)
            return tokenizer
        except Exception as e:
            raise CustomException(f"Failed to load tokenizer: {e}", sys)

    def evaluate_model(self, x_test, y_test):
        try:
            model = self.load_model()
            tokenizer = self.load_tokenizer()

            # Convert X_test to list of strings if needed
            x_test = [str(text) for text in x_test]

            sequences = tokenizer.texts_to_sequences(x_test)
            sequences_matrix = pad_sequences(sequences, maxlen=self.config.MAX_LEN)
            predictions = model.predict(sequences_matrix)
            predictions = (predictions > 0.5).astype("int32")

            # Assuming y_test is categorical, convert it to integer labels if needed
            if isinstance(y_test[0], np.ndarray):
                y_test = np.argmax(y_test, axis=1)

            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)

            logging.info(f"Accuracy: {accuracy}")
            logging.info(f"Precision: {precision}")
            logging.info(f"Recall: {recall}")
            logging.info(f"F1 Score: {f1}")

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
        except Exception as e:
            raise CustomException(f"Error during evaluation: {e}", sys)

# Main function to run the entire pipeline
if __name__ == "__main__":
    try:
        # Load dataset
        df = pd.read_csv(Config.CLEANED_DATA_PATH)

        # Clean data
        data_cleaner = DataCleaning()
        df = data_cleaner.clean_data(df, 'tweet')  # Ensure 'tweet' is the correct column name

        # Tokenize texts
        text_tokenizer = TextTokenization()
        text_tokenizer.fit_tokenizer(df['tweet'].values)
        sequences_matrix = text_tokenizer.transform_texts(df['tweet'].values)

        # Load test data for evaluation
        X_test = np.load(os.path.join('notebook', 'X_test.npy'), allow_pickle=True)
        y_test = np.load(os.path.join('notebook', 'y_test.npy'), allow_pickle=True)

        # Evaluate model
        evaluator = ModelEvaluation(Config)
        results = evaluator.evaluate_model(X_test.tolist(), y_test)
        print(results)

    except Exception as e:
        logging.error(f"Error in the pipeline: {e}")
        raise CustomException(f"Pipeline error: {e}", sys)
