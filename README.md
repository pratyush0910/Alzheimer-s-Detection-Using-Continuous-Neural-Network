# Alzheimer's Disease Detection from MRI Scans
This project is an end-to-end machine learning application designed to classify brain MRI scans into two categories: Cognitively Normal (CN) and Alzheimer's Disease (AD). The project utilizes a custom deep learning model featuring a ContinuousLayer and is deployed as a user-friendly web application using Flask.

The entire workflow, from data ingestion to model training and evaluation, is managed and versioned using DVC, ensuring full reproducibility.

## Project Workflow
This project is structured as a modular, multi-stage pipeline. Each stage is self-contained and can be reproduced independently.

**Data Ingestion:**

Downloads the dataset from a specified source (Google Drive).

Extracts the raw image files.

**Data Transformation:**

Loads the raw images and their labels (AD, CN).

Balances the dataset by down-sampling the majority class to prevent model bias.

Resizes all images to a consistent size (224x224).

Splits the data into training and testing sets.

**Base Model Preparation:** 

Builds the custom Keras model architecture, which includes the novel ContinuousLayer.

Saves the uncompiled model structure as an artifact.

**Model Training:**

Loads the prepared base model.

Compiles the model with the custom VariationalLoss and Adam optimizer.

Sets up data generators with optional augmentation for training.

Trains the model and saves the final, trained model weights.

**Model Evaluation:**

Loads the trained model.

Evaluates its performance on the unseen test set.

Calculates and saves key metrics like loss and accuracy.

Generates a classification report and a confusion matrix for detailed performance analysis.

**Prediction & Deployment:**

A Flask web application provides a simple user interface.

Users can upload an MRI scan.

The application uses a dedicated prediction component to load the model, preprocess the image, and return a prediction with class probabilities.

## Sample Output
The final output is a clean web interface where a user can upload an MRI image and receive a classification. The result displays the predicted class, the probabilities for both "Normal" and "Alzheimer's", and shows the uploaded image.

## Installation and Setup Guide
Follow these steps to set up and run the project locally.

**Prerequisites**
Git

Python 3.8+

A virtual environment tool (like venv or conda)

**1. Clone the Repository**
git clone https://github.com/your-username/Your-Repo-Name.git
```
cd Your-Repo-Name
```

**2. Create a Virtual Environment**
It's highly recommended to use a virtual environment to manage project dependencies.

 Using venv (recommended)
```
python -m venv tf_env
.\tf_env\Scripts\activate
```

**3. Install Dependencies** 
Install all the required Python packages from the requirements.txt file.
```
pip install -r requirements.txt
```
**4. Install the Local Package**
To make the alzheimer_classifier source code available for import, install it in editable mode.
```
pip install -e .
```
How to Run the Project
There are two main ways to run this project: running the training pipeline or running the web application for prediction.

**1. Running the Training Pipeline**
This project uses DVC to manage the end-to-end training pipeline.

**Initialize DVC (only once):**
```
dvc init
```
**Run the full pipeline:**
This command will execute all stages defined in dvc.yaml (data ingestion, transformation, training, and evaluation), skipping any stages that are already up-to-date.
```
dvc repro
```
**2. Running the Web Application**
To use the trained model for predictions, run the Flask web application.
```
python app.py
```
After running the command, open your web browser and navigate to:
http://127.0.0.1:8080

You will see the web interface where you can upload an MRI image to get a classification.

## Future Enhancements
This project provides a solid foundation that can be extended in several ways:

**Multi-Class Classification:** Extend the model to classify intermediate stages of cognitive decline, such as Mild Cognitive Impairment (MCI).

**Experiment Tracking:** Integrate tools like MLflow or Weights & Biases to log experiments, track metrics, and manage model versions systematically.

**Cloud Deployment:** Deploy the Flask application to a cloud service like AWS, Google Cloud, or Heroku to make it publicly accessible.

**Data Versioning with DVC:** Set up a DVC remote (e.g., Google Drive, S3) to version control the dataset and model artifacts, allowing for full reproducibility and collaboration.
