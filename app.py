from flask import Flask, request, render_template, url_for
import os
from werkzeug.utils import secure_filename
from alzheimer_classifier.components.prediction import Prediction
from alzheimer_classifier import logger

# Initialize the Flask app
app = Flask(__name__)

# Define the path for uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET'])
def home():
    # Render the main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        logger.error("No image file part in the request.")
        return render_template('index.html', error='No image file provided.')
        
    imagefile = request.files['image']
    
    if not imagefile or not imagefile.filename:
        logger.error("No image selected or filename is empty.")
        return render_template('index.html', error='No image selected.')

    try:
        # Secure the filename to prevent malicious file names
        filename = secure_filename(imagefile.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        imagefile.save(filepath)
        logger.info(f"Image saved to {filepath}")

        # Use the Prediction component to get the result
        prediction_component = Prediction(filepath)
        prediction_result = prediction_component.get_prediction()
        logger.info(f"Prediction for {filename}: {prediction_result}")
        
        # Pass the image path and the result dictionary to the template
        image_url = url_for('static', filename=f'uploads/{filename}')
        return render_template('index.html', result=prediction_result, image_path=image_url)
        
    except Exception as e:
        logger.exception(f"Prediction failed for {imagefile.filename}")
        return render_template('index.html', error=f'An error occurred during prediction: {e}')

if __name__ == "__main__":
    # Run the app on localhost, port 8080
    app.run(host="0.0.0.0", port=8080, debug=True)
