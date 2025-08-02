import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from alzheimer_classifier import logger
from alzheimer_classifier.entity.config_entity import TrainingConfig
from alzheimer_classifier.components.prepare_base_model import ContinuousLayer
from keras.callbacks import ModelCheckpoint

class VariationalLoss(keras.losses.Loss):
    """
    Custom loss function that combines Binary Cross-Entropy with a
    smoothness penalty for the ContinuousLayer kernels.
    """
    def __init__(self, model, lambda1=0.01, lambda2=1.0, **kwargs):
        super(VariationalLoss, self).__init__(**kwargs)
        self.model = model
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.bce = keras.losses.BinaryCrossentropy()

    def call(self, y_true, y_pred):
        smoothness_penalty = tf.constant(0.0, dtype=tf.float32)
        for layer in self.model.layers:
            if isinstance(layer, ContinuousLayer):
                smoothness_penalty += layer.smoothness_penalty()
        
        prediction_loss = self.bce(y_true, y_pred)
        
        total_loss = (self.lambda2 * prediction_loss) + (self.lambda1 * smoothness_penalty)
        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "lambda1": self.lambda1,
            "lambda2": self.lambda2
        })
        return config

class Training:
    """
    Handles the model training process, replicating the Colab notebook's methodology.
    """
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.train_generator = None
        self.valid_generator = None

    def load_base_model(self):
        """
        Loads the uncompiled base model from the previous stage.
        """
        logger.info("Loading base model...")
        custom_objects = {"ContinuousLayer": ContinuousLayer}
        self.model = keras.models.load_model(
            self.config.updated_base_model_path,
            custom_objects=custom_objects,
            compile=False
        )
        logger.info("Base model loaded successfully.")

    def _prepare_data(self):
        """
        Loads, balances, and splits the data into train, validation, and test sets
        using pandas DataFrames, as done in the notebook.
        """
        logger.info("Preparing data from scratch...")
        
        # 1. Load all image paths into a DataFrame
        image_paths = []
        labels = []
        categories = os.listdir(self.config.training_data)
        
        for category in categories:
            category_path = os.path.join(self.config.training_data, category)
            if os.path.isdir(category_path):
                for image_name in os.listdir(category_path):
                    image_paths.append(os.path.join(category_path, image_name))
                    labels.append(category)

        df = pd.DataFrame({"image_path": image_paths, "label": labels})

        # 2. Encode labels
        label_encoder = LabelEncoder()
        df['category_encoded'] = label_encoder.fit_transform(df['label'])
        df['category_encoded'] = df['category_encoded'].astype(str) # For flow_from_dataframe

        # 3. Balance the dataset (undersampling)
        min_samples = df['category_encoded'].value_counts().min()
        balanced_df = df.groupby('category_encoded').sample(n=min_samples, random_state=42)
        balanced_df = balanced_df.reset_index(drop=True)
        
        logger.info(f"Balanced dataset created with {min_samples} samples per class.")

        # 4. Split the data
        train_df, temp_df = train_test_split(
            balanced_df,
            train_size=0.8,
            shuffle=True,
            random_state=42,
            stratify=balanced_df['category_encoded']
        )
        
        valid_df, _ = train_test_split( # Test set is not used in training script
            temp_df,
            test_size=0.5,
            shuffle=True,
            random_state=42,
            stratify=temp_df['category_encoded']
        )
        
        return train_df, valid_df

    def _setup_data_generators(self, train_df, valid_df):
        """
        Sets up training and validation data generators with simple rescaling.
        """
        logger.info("Setting up data generators...")
        
        # Only rescaling is applied, as in the notebook
        train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        valid_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        self.train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col='image_path',
            y_col='category_encoded',
            target_size=self.config.params_image_size[:-1],
            class_mode='binary',
            batch_size=self.config.params_batch_size,
            shuffle=True
        )

        self.valid_generator = valid_datagen.flow_from_dataframe(
            dataframe=valid_df,
            x_col='image_path',
            y_col='category_encoded',
            target_size=self.config.params_image_size[:-1],
            class_mode='binary',
            batch_size=self.config.params_batch_size,
            shuffle=True # Validation can be shuffled as it's not time-series
        )
        logger.info("Data generators created successfully.")

    def train(self):
        """
        Compiles the model and runs the training process.
        """
        if not self.model:
            logger.error("Model is not loaded. Please call load_base_model() first.")
            return

        # Prepare data (load, balance, split)
        train_df, valid_df = self._prepare_data()

        # Setup data generators from the prepared dataframes
        self._setup_data_generators(train_df, valid_df)

        # Compile the model
        logger.info("Compiling the model with VariationalLoss...")
        loss = VariationalLoss(model=self.model, lambda1=0.01, lambda2=1.0)
        optimizer = keras.optimizers.Adam(learning_rate=self.config.params_learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        logger.info("Model compiled successfully.")

        # Setup ModelCheckpoint to save the best model
        checkpoint = ModelCheckpoint(
            filepath=self.config.trained_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        logger.info("Starting model training...")
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            verbose=1, # To see the progress
            callbacks=[checkpoint]
        )
        logger.info("Model training completed.")
