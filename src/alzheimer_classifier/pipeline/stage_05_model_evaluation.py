from alzheimer_classifier.config.configuration import ConfigurationManager
from alzheimer_classifier.components.model_evaluation import ModelEvaluation
from alzheimer_classifier import logger

STAGE_NAME = "Stage 05: Model Evaluation"

class ModelEvaluationPipeline:
    """
    Orchestrates the model evaluation process.
    """
    def __init__(self):
        pass

    def main(self):
        """
        Executes the main steps of the model evaluation pipeline.
        """
        try:
            logger.info(f"--- Starting {STAGE_NAME} ---")
            config = ConfigurationManager()
            evaluation_config = config.get_evaluation_config()
            evaluation = ModelEvaluation(config=evaluation_config)
            evaluation.evaluate_and_save_metrics()
            logger.info(f"--- {STAGE_NAME} completed successfully ---\n")
        except Exception as e:
            logger.exception(f"Exception occurred during {STAGE_NAME}")
            raise e
