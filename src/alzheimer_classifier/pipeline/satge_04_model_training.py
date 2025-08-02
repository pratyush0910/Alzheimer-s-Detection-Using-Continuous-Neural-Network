# src/alzheimer_classifier/pipeline/stage_03_model_training.py
from alzheimer_classifier.config.configuration import ConfigurationManager
from alzheimer_classifier.components.prepare_base_model import PrepareBaseModel
from alzheimer_classifier.components.model_training import Training
from alzheimer_classifier import logger

STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info(f"--- Starting {STAGE_NAME} ---")
        # Train Model
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.load_base_model()
        training.train()
        logger.info(f"--- {STAGE_NAME} completed successfully ---\n")

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e