from alzheimer_classifier.config.configuration import ConfigurationManager
from alzheimer_classifier.components.prepare_base_model import PrepareBaseModel
from alzheimer_classifier import logger

STAGE_NAME = "Stage 03: Prepare Base Model"

class PrepareBaseModelTrainingPipeline:
    """
    Orchestrates the base model preparation process.
    """
    def __init__(self):
        pass

    def main(self):
        """
        Executes the main steps of the base model preparation pipeline.
        """
        try:
            logger.info(f"--- Starting {STAGE_NAME} ---")
            config = ConfigurationManager()
            prepare_base_model_config = config.get_prepare_base_model_config()
            prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
            prepare_base_model.prepare_and_save_model()
            logger.info(f"--- {STAGE_NAME} completed successfully ---\n")
        except Exception as e:
            logger.exception(f"Exception occurred during {STAGE_NAME}")
            raise e
if __name__ == "__main__":
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()        