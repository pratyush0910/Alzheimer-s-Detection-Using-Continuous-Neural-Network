import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from alzheimer_classifier import logger
from alzheimer_classifier.entity.config_entity import PrepareBaseModelConfig

class ContinuousLayer(keras.layers.Layer):
    """
    Custom Keras layer implementing a continuous kernel convolution,
    matching the implementation in the reference notebook.
    """
    def __init__(self, kernel_size=5, num_basis=10, output_channels=16, **kwargs):
        super(ContinuousLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_basis = num_basis
        self.output_channels = output_channels
        self.centers = self.add_weight(name='centers', shape=(num_basis, 2), initializer='random_normal', trainable=True)
        self.widths = self.add_weight(name='widths', shape=(num_basis,), initializer='ones', trainable=True, constraint=keras.constraints.NonNeg())
        # Note: The kernel_weights shape is different from your original implementation to match the notebook's logic.
        self.kernel_weights = self.add_weight(
            name='kernel_weights',
            shape=(kernel_size, kernel_size, 3, output_channels), # Input channels = 3 for RGB
            initializer='glorot_normal',
            trainable=True
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "num_basis": self.num_basis,
            "output_channels": self.output_channels,
        })
        return config

    def call(self, inputs):
        shape = tf.shape(inputs)
        height, width = shape[1], shape[2]

        x = tf.range(0, tf.cast(height, tf.float32), 1.0) #type: ignore
        y = tf.range(0, tf.cast(width, tf.float32), 1.0) #type: ignore

        x_grid, y_grid = tf.meshgrid(x, y)
        grid = tf.stack([x_grid, y_grid], axis=-1)

        basis = []
        for i in range(self.num_basis):
            center = self.centers[i]
            width_val = self.widths[i]
            dist = tf.reduce_sum(((grid - center) / width_val) ** 2, axis=-1)
            basis_i = tf.exp(-dist)
            basis.append(basis_i)
        basis = tf.stack(basis, axis=-1)

        basis_weights = tf.reduce_mean(basis, axis=[0, 1])
        basis_weights = tf.nn.softmax(basis_weights)
        
        # The modulation logic is adjusted to match the notebook
        reshaped_weights = tf.reshape(basis_weights, [self.num_basis, 1, 1, 1, 1])
        
        # The kernel_weights are expanded to be modulated by basis functions
        expanded_kernel_weights = tf.expand_dims(self.kernel_weights, axis=0)
        
        modulated_kernel = tf.reduce_sum(reshaped_weights * expanded_kernel_weights, axis=0)

        return tf.nn.conv2d(inputs, modulated_kernel, strides=[1, 1, 1, 1], padding='SAME')

    def smoothness_penalty(self):
        # This penalty encourages the kernel weights to be smooth.
        grad_x = tf.reduce_mean(tf.square(self.kernel_weights[1:, :, :, :] - self.kernel_weights[:-1, :, :, :]))
        grad_y = tf.reduce_mean(tf.square(self.kernel_weights[:, 1:, :, :] - self.kernel_weights[:, :-1, :, :]))
        return grad_x + grad_y

class PrepareBaseModel:
    """
    Builds the Keras model architecture and saves it as an uncompiled model.
    """
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None

    def _build_model(self):
        """
        Constructs the model architecture using the ContinuousLayer, same as the notebook.
        """
        logger.info("Building the base model architecture...")
        inputs = keras.Input(shape=self.config.params_image_size)
        
        x = ContinuousLayer(kernel_size=5, num_basis=10, output_channels=16)(inputs)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        outputs = keras.layers.Dense(1, activation='sigmoid') (x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.summary(print_fn=logger.info)
        logger.info("Base model architecture built successfully.")

    def save_model(self):
        """
        Saves the uncompiled model to the path specified in the config.
        """
        if self.model:
            logger.info(f"Saving uncompiled base model to: {self.config.updated_base_model_path}")
            self.model.save(self.config.updated_base_model_path)
            logger.info("Uncompiled base model saved successfully.")
        else:
            logger.error("Model has not been built. Cannot save.")

    def prepare_and_save_model(self):
        """
        Main orchestrator method for this component.
        """
        self._build_model()
        self.save_model()
