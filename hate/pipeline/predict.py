import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from config.config import Config

# Load the trained model and tokenizer
model = load_model(Config.MODEL_PATH)
tokenizer = joblib.load(Config.TOKENIZER_PATH)

def preprocess_text(text):
    """
    Preprocess the input text by tokenizing and padding sequences.
    """
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=Config.MAX_LEN)
    return padded_sequences

def predict(text):
    """
    Predict the class of the given text.
    """
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)
    pred = prediction[0][0]
    print("pred", pred)
    if pred > 0.5:
        print("hate and abusive")
        return "hate and abusive"
    else:
        print("no hate")
        return "no hate"

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_text = sys.argv[1]
        result = predict(input_text)
        print(f"Input: {input_text}")
        print(f"Prediction: {result}")
    else:
        print("Please provide text to predict.")
