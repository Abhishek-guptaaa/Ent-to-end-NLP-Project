import pandas as pd
from keras.models import load_model
from keras.utils import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config.config import Config
from hate.logger import logging
from hate.exception import CustomException
import joblib

class ModelEvaluation:
    def __init__(self):
        self.model = None

    def load_model(self):
        try:
            self.model = load_model(Config.MODEL_PATH)
            logging.info(f"Model loaded from {Config.MODEL_PATH}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise CustomException("Error loading model", e)

    def preprocess_data(self, data):
        # Implement preprocessing steps similar to training preprocessing
        # Convert all text data to string to avoid 'float' object error
        data = data.astype(str)
        return data

    def evaluate_model(self):
        try:
            logging.info("Starting model evaluation")

            # Load the test data from the CSV file
            test_data = pd.read_csv(Config.X_TEST_PATH)
            X_test = test_data[Config.TWEET].astype(str)  #Convert to string to avoid 'float' object error
            y_test = test_data[Config.CLASS]  # Adjust the column name

            # Ensure the test data is preprocessed in the same way as training data
            X_test = self.preprocess_data(X_test)  # Implement this method

            # Load tokenizer and transform the test data
            tokenizer = joblib.load(Config.TOKENIZER_PATH)
            X_test = tokenizer.texts_to_sequences(X_test)
            X_test = pad_sequences(X_test, maxlen=Config.MAX_LEN)

            # Evaluate the model
            y_pred = self.model.predict(X_test)
            y_pred = (y_pred > 0.5).astype(int)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logging.info(f"Model evaluation completed. Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

            return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
        except Exception as e:
            logging.error(f"Error in evaluating model: {e}")
            raise CustomException("Error in evaluating model", e)

if __name__ == "__main__":
    # Initialize ModelEvaluation instance
    evaluator = ModelEvaluation()

    try:
        # Load the trained model
        evaluator.load_model()

        # Perform model evaluation
        evaluation_results = evaluator.evaluate_model()
        logging.info(f"Model evaluation results: {evaluation_results}")

    except CustomException as ce:
        logging.error(f"CustomException occurred: {ce}")
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}")