# import sys
# import os
# import joblib
# import pandas as pd
# from config.config import Config
# from hate.logger import logging
# from hate.components.data_ingestion import DataIngestion
# from hate.components.data_cleaning import DataCleaning
# from hate.components.text_tokenization import TextTokenization
# from hate.components.model_training import ModelTraining
# from hate.components.model_evaluation import ModelEvaluation
# from hate.exception import CustomException

# def main():
#     try:
#         logging.info("Starting NLP training pipeline")

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
        
#         # Save the tokenizer
#         joblib.dump(text_tokenization.tokenizer, Config.TOKENIZER_PATH)

#         # Model Training
#         model_training = ModelTraining(sequences_matrix, cleaned_data[Config.CLASS])
#         model_training.build_model()
#         model_training.train_model()

#         # Model Evaluation
#         model_evaluation = ModelEvaluation()
#         model_evaluation.load_model()  # Ensure model is loaded
#         evaluation_results = model_evaluation.evaluate_model()
#         logging.info(f"Model evaluation results: {evaluation_results}")

#         logging.info("NLP training pipeline completed successfully")
#     except Exception as e:
#         logging.error(f"Error in main script: {e}")
#         raise CustomException(e, sys)

# if __name__ == "__main__":
#     main()



import sys
import os
import joblib
import pandas as pd
from config.config import Config
from hate.logger import logging
from hate.components.data_ingestion import DataIngestion
from hate.components.data_cleaning import DataCleaning
from hate.components.text_tokenization import TextTokenization
from hate.components.model_training import ModelTraining
from hate.components.model_evaluation import ModelEvaluation
from hate.exception import CustomException

class TrainPipeline:
    def run_pipeline(self):
        try:
            logging.info("Starting NLP training pipeline")

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
            
            # Save the tokenizer
            joblib.dump(text_tokenization.tokenizer, Config.TOKENIZER_PATH)

            # Model Training
            model_training = ModelTraining(sequences_matrix, cleaned_data[Config.CLASS])
            model_training.build_model()
            model_training.train_model()

            # Model Evaluation
            model_evaluation = ModelEvaluation()
            model_evaluation.load_model()  # Ensure model is loaded
            evaluation_results = model_evaluation.evaluate_model()
            logging.info(f"Model evaluation results: {evaluation_results}")

            logging.info("NLP training pipeline completed successfully")
        except Exception as e:
            logging.error(f"Error in main script: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    TrainPipeline().run_pipeline()
