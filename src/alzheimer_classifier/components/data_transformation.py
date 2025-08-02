import os
import pandas as pd
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from alzheimer_classifier import logger
from alzheimer_classifier.entity.config_entity import DataTransformationConfig

class DataTransformation:
    """
    Handles the transformation of the raw dataset into a clean,
    structured format ready for training. This version includes
    enhanced logging to diagnose path and file issues.
    """
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.df = pd.DataFrame()

    def _load_data_to_dataframe(self):
        """Loads all image paths and their corresponding labels into a pandas DataFrame."""
        logger.info("--- Starting Data Loading for Transformation ---")
        logger.info(f"Attempting to read data from base path: {self.config.data_path}")

        # CRITICAL CHECK: Does the input directory from Stage 1 exist?
        if not os.path.exists(self.config.data_path):
            logger.error(f"FATAL ERROR: The input data path does not exist: {self.config.data_path}")
            logger.error("Please check the 'data_path' in your 'config.yaml' under the 'data_transformation' section.")
            logger.error(f"Verify that the unzipping process in Stage 1 created this folder.")
            return

        image_paths = []
        labels = []
        categories = ["AD", "CN"]
        
        for category in categories:
            category_path = self.config.data_path / category
            logger.info(f"Checking for category directory: {category_path}")

            if not os.path.isdir(category_path):
                logger.warning(f"Directory not found for category '{category}'. It will be skipped. Path: {category_path}")
                continue
            
            images_found = os.listdir(category_path)
            if not images_found:
                logger.warning(f"No images found in directory for category '{category}'. Path: {category_path}")
                continue

            logger.info(f"Found {len(images_found)} images in '{category}' directory.")
            for image_name in images_found:
                image_paths.append(category_path / image_name)
                labels.append(category)
        
        if not image_paths:
            logger.error("FATAL ERROR: No images were loaded from any category.")
            logger.error(f"Please ensure that the subdirectories 'AD' and 'CN' exist inside '{self.config.data_path}' and that they contain image files.")
            return

        self.df = pd.DataFrame({"image_path": image_paths, "label": labels})
        logger.info(f"Successfully loaded a total of {len(self.df)} images into the DataFrame.")
        logger.info("--- Finished Data Loading ---")

    def _balance_data(self):
        """Balances the dataset by down-sampling the majority class."""
        if self.df.empty:
            logger.error("DataFrame is empty. Skipping balancing step.")
            return

        logger.info("Balancing the dataset by down-sampling...")
        min_samples = self.df['label'].value_counts().min()
        balanced_df = self.df.groupby('label').sample(n=min_samples, random_state=42)
        self.df = balanced_df.reset_index(drop=True)
        logger.info(f"Dataset balanced. New total size: {len(self.df)} images.")

    def _split_and_save_data(self):
        """Splits the data into training and testing sets and saves them."""
        if self.df.empty:
            logger.error("DataFrame is empty. Skipping split and save step.")
            return

        logger.info("Splitting data into training and testing sets.")
        train_df, test_df = train_test_split(
            self.df, 
            test_size=0.2, 
            random_state=42, 
            stratify=self.df['label']
        )

        logger.info(f"Resizing and saving {len(train_df)} training images...")
        self._save_images(train_df, self.config.train_dir)
        
        logger.info(f"Resizing and saving {len(test_df)} testing images...")
        self._save_images(test_df, self.config.test_dir)

    def _save_images(self, dataframe: pd.DataFrame, target_dir: Path):
        """Resizes and saves images from the dataframe to the target directory."""
        os.makedirs(target_dir, exist_ok=True)
        logger.info(f"Saving images to '{target_dir}'...")
        for _, row in dataframe.iterrows():
            label = row['label']
            image_path = Path(row['image_path'])
            class_dir = target_dir / label
            os.makedirs(class_dir, exist_ok=True)

            try:
                img = cv2.imread(str(image_path))
                if img is None:
                    logger.warning(f"Could not read image file: {image_path}. Skipping.")
                    continue
                img_resized = cv2.resize(img, (224, 224))
                save_path = class_dir / image_path.name
                cv2.imwrite(str(save_path), img_resized)
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
        logger.info(f"Finished saving images to '{target_dir}'.")

    def process_and_split_data(self):
        """Main method to orchestrate the entire transformation process."""
        self._load_data_to_dataframe()
        if self.df.empty:
            logger.error("Halting data transformation because no data was loaded.")
            return
        self._balance_data()
        self._split_and_save_data()
        logger.info("Data transformation stage finished.")
