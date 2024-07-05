
import joblib
from keras.models import load_model
from keras.utils import pad_sequences
from config.config import Config

class PredictionPipeline:
    def __init__(self):
        # Load the trained model and tokenizer
        self.model = load_model(Config.MODEL_PATH)
        self.tokenizer = joblib.load(Config.TOKENIZER_PATH)

    def preprocess_text(self, text):
        """
        Preprocess the input text by tokenizing and padding sequences.
        """
        sequences = self.tokenizer.texts_to_sequences([text])
        padded_sequences = pad_sequences(sequences, maxlen=Config.MAX_LEN)
        print(f"Tokenized and Padded Text: {padded_sequences}")  # Debugging step
        return padded_sequences

    def predict(self, text):
        """
        Predict the class of the given text.
        """
        processed_text = self.preprocess_text(text)
        prediction = self.model.predict(processed_text)
        pred = prediction[0][0]
        print("Prediction Score:", pred)  # Debugging step
        if pred > 0.3:
            print("hate and abusive")
            return "hate and abusive"
        else:
            print("no hate")
            return "no hate"
