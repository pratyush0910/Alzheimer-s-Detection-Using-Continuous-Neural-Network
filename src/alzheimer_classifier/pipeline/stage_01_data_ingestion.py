from alzheimer_classifier.config.configuration import ConfigurationManager
from alzheimer_classifier.components.data_ingestion import DataIngestion
from alzheimer_classifier import logger

STAGE_NAME = "Stage 01: Data Ingestion"

class DataIngestionTrainingPipeline:
    """
    Orchestrates the data ingestion process.
    """
    def __init__(self):
        pass

    def main(self):
        """
        Executes the main steps of the data ingestion pipeline.
        """
        try:
            logger.info(f"--- Starting {STAGE_NAME} ---")
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()
            logger.info(f"--- {STAGE_NAME} completed successfully ---\n")
        except Exception as e:
            logger.exception(f"Exception occurred during {STAGE_NAME}")
            raise e
        if __name__ == "__main__":
         ingestion = DataIngestionTrainingPipeline()
         ingestion.main()

    # Add this:
         import os
         print("Expected unzip path:", os.path.abspath("artifacts/data_ingestion/Alzheimer_Dataset"))
         print("Does it exist?", os.path.exists("artifacts/data_ingestion/Alzheimer_Dataset"))
