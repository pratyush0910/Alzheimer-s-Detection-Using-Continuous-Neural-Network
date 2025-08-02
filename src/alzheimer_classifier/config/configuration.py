# src/alzheimer_classifier/config/configuration.py

from alzheimer_classifier.constants import *
import os
from pathlib import Path
from alzheimer_classifier.utils.common import read_yaml, create_directories
from alzheimer_classifier.entity.config_entity import (
    DataIngestionConfig, 
    DataTransformationConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig
)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )

        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])
        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            train_dir=Path(config.train_dir),
            test_dir=Path(config.test_dir)
        )
     
   
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params
        create_directories([config.root_dir])
        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,
            params_include_top=False,
            params_weights='imagenet',
            params_classes=params.NUM_CLASSES
        )
       

    def get_training_config(self) -> TrainingConfig:
        """
        Retrieves the model training configuration.
        """
        config = self.config.model_training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        
        training_data = config.training_data
        testing_data = config.testing_data
        
        create_directories([config.root_dir])

        training_config = TrainingConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            testing_data=Path(testing_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE
        ) 
        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        """
        Retrieves the model evaluation configuration.
        """
        eval_config = self.config.model_evaluation
        
        create_directories([eval_config.root_dir])

        evaluation_config = EvaluationConfig(
            path_of_model=eval_config.trained_model_path,
            testing_data=self.config.data_transformation.test_dir,
            metrics_path=eval_config.metrics_path,
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return evaluation_config
