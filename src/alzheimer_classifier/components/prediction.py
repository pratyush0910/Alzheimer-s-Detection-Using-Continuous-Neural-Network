import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import cv2
from alzheimer_classifier.components.prepare_base_model import ContinuousLayer

class Prediction:
    """
    This component is responsible for making a prediction on a single image.
    It re-creates the model architecture and loads the trained weights to
    avoid version conflicts during deployment.
    """
    def __init__(self, filename):
        self.filename = filename
        # Path to your trained model's weights
        self.model_weights_path = Path("artifacts/model_training/model.h5")

    def _create_model_architecture(self):
        """
        Re-creates the exact same model architecture that was used for training.
        """
        inputs = keras.Input(shape=(224, 224, 3))
        x = ContinuousLayer(kernel_size=5, num_basis=10, output_channels=16)(inputs)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def get_prediction(self):
        """
        Creates the model, loads weights, preprocesses the image, and returns a detailed prediction.
        """
        # Create a fresh model instance
        model = self._create_model_architecture()
        
        # Load only the trained weights into the model
        model.load_weights(self.model_weights_path)

        # Load and preprocess the image
        img = cv2.imread(self.filename)
        if img is None:
            return {"error": "Could not read image file."}
            
        img_resized = cv2.resize(img, (224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make a prediction
        prediction_score = model.predict(img_array)[0][0]
        
        # Calculate probabilities for both classes
        prob_ad = prediction_score * 100
        prob_normal = (1 - prediction_score) * 100
        
        # Determine the final class
        if prediction_score > 0.5:
            predicted_class = "Normal"
        else:
            predicted_class = "Alzheimer's Detected"
            
        return {
            "predicted_class": predicted_class,
            "probabilities": {
                "Alzheimer's": f"{prob_normal:.2f}%",
                "Normal": f"{prob_ad:.2f}%"
            }
        }
