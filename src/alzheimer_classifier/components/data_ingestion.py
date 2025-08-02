import os
import zipfile
import shutil
import gdown
from pathlib import Path
from alzheimer_classifier import logger
from alzheimer_classifier.entity.config_entity import DataIngestionConfig
from alzheimer_classifier.config.configuration import ConfigurationManager
class DataIngestion:
    """
    Handles the downloading and extraction of the dataset.
    """
    def __init__(self, config: DataIngestionConfig):
        """
        Initializes the DataIngestion component with its configuration.
        """
        self.config = config

    def download_file(self):
    
     print("Checking if dataset exists at:", self.config.local_data_file)
     print("Download URL is:", self.config.source_URL)

     if not os.path.exists(self.config.local_data_file):
            logger.info("Starting download of dataset...")
            
            # --- PROXY FIX STARTS HERE ---
            # Store original proxy settings
            original_proxies = {
                'http': os.environ.get('HTTP_PROXY'),
                'https': os.environ.get('HTTPS_PROXY')
            }
            # Temporarily disable proxies for this download
            os.environ['HTTP_PROXY'] = ''
            os.environ['HTTPS_PROXY'] = ''
            logger.info("Temporarily disabled system proxies for download.")
            # --- PROXY FIX ENDS HERE ---

            try:
                gdown.download(self.config.source_URL, str(self.config.local_data_file), quiet=False)
                logger.info(f"Dataset downloaded successfully and saved to: {self.config.local_data_file}")
            except Exception as e:
                logger.error(f"An error occurred during download: {e}")
                raise e
            finally:
                # --- RESTORE PROXY SETTINGS ---
                # Restore original proxy settings to avoid side effects
                if original_proxies['http']:
                    os.environ['HTTP_PROXY'] = original_proxies['http']
                if original_proxies['https']:
                    os.environ['HTTPS_PROXY'] = original_proxies['https']
                logger.info("System proxies restored.")
                # --- RESTORE ENDS HERE ---
     else:
      logger.info(f"File already exists at: {self.config.local_data_file}. Skipping download.")



    def extract_zip_file(self):
        logger.info(f"Starting extraction of {self.config.local_data_file}...")
        
        temp_extract_path = "temp_extract"
        os.makedirs(temp_extract_path, exist_ok=True)
        
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
         zip_ref.extractall(temp_extract_path)
        
        if os.path.exists(self.config.unzip_dir):
         shutil.rmtree(self.config.unzip_dir)
         
        os.makedirs(self.config.unzip_dir, exist_ok=True)
        
        for item in os.listdir(temp_extract_path):
         src = os.path.join(temp_extract_path, item)
         dst = os.path.join(self.config.unzip_dir, item)
         shutil.move(src, dst)
        
        shutil.rmtree(temp_extract_path) 
        logger.info(f"Successfully extracted zip file to: {self.config.unzip_dir}") 

if __name__ == "__main__":
    
    config = ConfigurationManager().get_data_ingestion_config()
    ingestion = DataIngestion(config=config)
    ingestion.download_file()
    ingestion.extract_zip_file()        
        