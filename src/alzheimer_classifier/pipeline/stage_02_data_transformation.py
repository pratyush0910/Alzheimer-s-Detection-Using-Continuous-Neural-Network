from alzheimer_classifier.config.configuration import ConfigurationManager
from alzheimer_classifier.components.data_transformation import DataTransformation
from alzheimer_classifier import logger

STAGE_NAME = "Stage 02: Data Transformation"

class DataTransformationTrainingPipeline:
    """
    Orchestrates the data transformation process.
    """
    def __init__(self):
        pass

    def main(self):
        """
        Executes the main steps of the data transformation pipeline.
        """
        try:
            logger.info(f"--- Starting {STAGE_NAME} ---")
            config = ConfigurationManager()
            # THE FIX: Corrected the method name from get_data_transformation__config to get_data_transformation_config
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation.process_and_split_data()
            logger.info(f"--- {STAGE_NAME} completed successfully ---\n")
        except Exception as e:
            logger.exception(f"Exception occurred during {STAGE_NAME}")
            raise e
