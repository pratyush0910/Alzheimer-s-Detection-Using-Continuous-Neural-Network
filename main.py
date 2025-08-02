from alzheimer_classifier import logger
from alzheimer_classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from alzheimer_classifier.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from alzheimer_classifier.pipeline.stage_03_prepare_base_model import PrepareBaseModelTrainingPipeline
from alzheimer_classifier.pipeline.satge_04_model_training import ModelTrainingPipeline
from alzheimer_classifier.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
# This is the main entry point for executing the full ML training pipeline.
# It runs each stage of the pipeline in sequence.

if __name__ == '__main__':
    # --- Stage 1: Data Ingestion ---
    try:
        logger.info("="*20)
        logger.info(">>>>> Starting Stage 01: Data Ingestion <<<<<")
        data_ingestion_pipeline = DataIngestionTrainingPipeline()
        data_ingestion_pipeline.main()
        logger.info(">>>>> Stage 01: Data Ingestion completed successfully <<<<<\n")
    except Exception as e:
        logger.exception(f"Error occurred during Stage 01: Data Ingestion. Details: {e}")
        raise e

    # --- Stage 2: Data Transformation ---
    try:
        logger.info("="*20)
        logger.info(">>>>> Starting Stage 02: Data Transformation <<<<<")
        data_transformation_pipeline = DataTransformationTrainingPipeline()
        data_transformation_pipeline.main()
        logger.info(">>>>> Stage 02: Data Transformation completed successfully <<<<<\n")
    except Exception as e:
        logger.exception(f"Error occurred during Stage 02: Data Transformation. Details: {e}")
        raise e

    # --- Stage 3: Prepare Base Model ---
    try:
        logger.info("="*20)
        logger.info(">>>>> Starting Stage 03: Prepare Base Model <<<<<")
        prepare_base_model_pipeline = PrepareBaseModelTrainingPipeline()
        prepare_base_model_pipeline.main()
        logger.info(">>>>> Stage 03: Prepare Base Model completed successfully <<<<<\n")
    except Exception as e:
        logger.exception(f"Error occurred during Stage 03: Prepare Base Model. Details: {e}")
        raise e

    # --- Stage 4: Model Training ---
    try:
        logger.info("="*20)
        logger.info(">>>>> Starting Stage 04: Model Training <<<<<")
        model_training_pipeline = ModelTrainingPipeline()
        model_training_pipeline.main()
        logger.info(">>>>> Stage 04: Model Training completed successfully <<<<<\n")
    except Exception as e:
        logger.exception(f"Error occurred during Stage 04: Model Training. Details: {e}")
        raise e
    
     # --- Stage 5: Model Evaluation ---
    try:
        logger.info("="*20)
        logger.info(">>>>> Starting Stage 05: Model Evaluation <<<<<")
        model_evaluation_pipeline = ModelEvaluationPipeline()
        model_evaluation_pipeline.main()
        logger.info(">>>>> Stage 05: Model Evaluation completed successfully <<<<<\n")
    except Exception as e:
        logger.exception(f"Error occurred during Stage 05: Model Evaluation. Details: {e}")
        raise e
