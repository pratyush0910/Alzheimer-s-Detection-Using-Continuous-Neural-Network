import tensorflow as tf
from pathlib import Path
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from alzheimer_classifier import logger
from alzheimer_classifier.entity.config_entity import EvaluationConfig
from alzheimer_classifier.components.prepare_base_model import ContinuousLayer
from alzheimer_classifier.components.model_training import VariationalLoss

class ModelEvaluation:
    """
    Handles the evaluation of the trained model on the test dataset.
    """
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.test_generator = None
        self._create_test_generator()

    def _create_test_generator(self):
        """
        Creates a data generator for the test set.
        """
        test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )

        self.test_generator = test_datagenerator.flow_from_directory(
            directory=str(self.config.testing_data),
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode='binary',
            shuffle=False
        )
        logger.info("Test data generator created.")

    def evaluate_and_save_metrics(self):
        """
        Evaluates the model on the test set and saves the metrics.
        """
        logger.info("Loading trained model for evaluation...")

        model = tf.keras.models.load_model(
            self.config.path_of_model,
            custom_objects={"ContinuousLayer": ContinuousLayer},
            compile=False
        )

        # THE FIX: Check if the model was loaded successfully *before* trying to use it.
        if not model:
            logger.error("Model could not be loaded. Aborting evaluation.")
            return

        logger.info("Re-compiling model with custom loss for evaluation...")
        loss = VariationalLoss(model=model)
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
            
        if not self.test_generator:
            logger.error("Test generator is not available. Aborting evaluation.")
            return

        logger.info("Evaluating model on the test set...")
        score = model.evaluate(self.test_generator)
        scores = {"loss": score[0], "accuracy": score[1]}
        
        logger.info(f"Test Loss: {scores['loss']:.4f}")
        logger.info(f"Test Accuracy: {scores['accuracy']:.4f}")

        with open(self.config.metrics_path, 'w') as f:
            json.dump(scores, f, indent=4)
        logger.info(f"Evaluation scores saved to: {self.config.metrics_path}")

        logger.info("Generating detailed classification report and confusion matrix...")
        self.test_generator.reset()
        y_pred = model.predict(self.test_generator)
        
        if y_pred is None:
            logger.error("Model prediction returned None. Aborting report generation.")
            return
            
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
        y_true = self.test_generator.classes
        class_names = list(self.test_generator.class_indices.keys())

        if y_true is None or len(y_true) != len(y_pred_binary):
            logger.error("Mismatch between true labels and predicted labels. Aborting report generation.")
            return

        report = classification_report(y_true, y_pred_binary, target_names=class_names)
        logger.info(f"Classification Report:\n{report}")

        cm = confusion_matrix(y_true, y_pred_binary)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        plot_path = Path(self.config.metrics_path).parent / "confusion_matrix.png"
        plt.savefig(plot_path)
        logger.info(f"Confusion matrix plot saved to: {plot_path}")
