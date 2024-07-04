import sys
import os
import pandas as pd
from config.config import Config
from hate.logger import logging
from hate.components.data_ingestion import DataIngestion
from hate.components.data_cleaning import DataCleaning
from hate.components.text_tokenization import TextTokenization
from hate.components.model_training import ModelTraining
from hate.exception import CustomException
from hate.components.model_evaluation import  ModelEvaluation

def main():
    try:
        logging.info("Starting NLP project")

        # Data Ingestion
        data_ingestion = DataIngestion()
        raw_data_path = data_ingestion.initiate_data_ingestion()
        
        # Load the raw data
        raw_data = pd.read_csv(raw_data_path)
        
        # Ensure raw_data is a DataFrame
        if not isinstance(raw_data, pd.DataFrame):
            raise CustomException("Data ingestion did not return a DataFrame", sys)
        
        # Data Cleaning 
        data_cleaning = DataCleaning()
        cleaned_data = data_cleaning.clean_data(raw_data, Config.TWEET)
        
        # Ensure the directory for the cleaned data file exists
        os.makedirs(os.path.dirname(Config.CLEANED_DATA_PATH), exist_ok=True)

        # Save cleaned data to CSV
        cleaned_data.to_csv(Config.CLEANED_DATA_PATH, index=False)
        
        # Text Tokenization 
        text_tokenization = TextTokenization()
        text_tokenization.fit_tokenizer(cleaned_data[Config.TWEET])
        sequences_matrix = text_tokenization.transform_texts(cleaned_data[Config.TWEET])
        
        # Model Training
        model_training = ModelTraining(sequences_matrix, cleaned_data[Config.CLASS])
        model_training.build_model()
        model_training.train_model()

        evaluation_results = ModelEvaluation.evaluate_model()
        logging.info(f"Model evaluation results: {evaluation_results}")

        logging.info("NLP project completed successfully")
    except Exception as e:
        logging.error(f"Error in main script: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()







# import os
# import sys
# import pandas as pd
# from config.config import Config
# from hate.logger import logging
# from hate.components.data_ingestion import DataIngestion
# from hate.components.data_cleaning import DataCleaning
# from hate.components.text_tokenization import TextTokenization
# from hate.components.model_training import ModelTraining
# from hate.components.model_evaluation import ModelEvaluation
# from hate.exception import CustomException
# from keras.utils import pad_sequences
# import joblib

# def main():
#     try:
#         logging.info("Starting NLP project")

#         # Data Ingestion
#         data_ingestion = DataIngestion()
#         raw_data_path = data_ingestion.initiate_data_ingestion()
        
#         # Load the raw data
#         raw_data = pd.read_csv(raw_data_path)
        
#         # Ensure raw_data is a DataFrame
#         if not isinstance(raw_data, pd.DataFrame):
#             raise CustomException("Data ingestion did not return a DataFrame", sys)
        
#         # Data Cleaning 
#         data_cleaning = DataCleaning()
#         cleaned_data = data_cleaning.clean_data(raw_data, Config.TWEET)
        
#         # Ensure the directory for the cleaned data file exists
#         os.makedirs(os.path.dirname(Config.CLEANED_DATA_PATH), exist_ok=True)

#         # Save cleaned data to CSV
#         cleaned_data.to_csv(Config.CLEANED_DATA_PATH, index=False)
        
#         # Text Tokenization 
#         text_tokenization = TextTokenization()
#         text_tokenization.fit_tokenizer(cleaned_data[Config.TWEET])
#         sequences_matrix = text_tokenization.transform_texts(cleaned_data[Config.TWEET])
        
#         # Model Training
#         model_training = ModelTraining(sequences_matrix, cleaned_data[Config.CLASS])
#         model_training.build_model()
#         history, (X_test, y_test) = model_training.train_model()

#         # Save the tokenizer
#         joblib.dump(text_tokenization.tokenizer, Config.TOKENIZER_PATH)
#         logging.info(f"Tokenizer saved at {Config.TOKENIZER_PATH}")

#         # Model Evaluation
#         model_evaluation = ModelEvaluation()
#         model_evaluation.load_model()
#         evaluation_results = model_evaluation.evaluate_model()
#         logging.info(f"Model evaluation results: {evaluation_results}")

#         logging.info("NLP project completed successfully")
#     except Exception as e:
#         logging.error(f"Error in main script: {e}")
#         raise CustomException(e, sys)

# if __name__ == "__main__":
#     main()





