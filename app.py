import os
import numpy as np
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image
import io
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
MODEL_PATH = 'vgg19_model_03.h5'
model = None

def load_pneumonia_model():
    global model
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Load model on startup
load_pneumonia_model()

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess the image for VGG19 model prediction"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")
        
        # Convert BGR to RGB (OpenCV loads in BGR, but most models expect RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224 (VGG19 input size)
        img = cv2.resize(img, (128, 128))
        
        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0  # Normalize to [0,1]
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_pneumonia(image_path):
    """Make prediction using the loaded model"""
    if model is None:
        return None, "Model not loaded"
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_path)
        if processed_image is None:
            return None, "Failed to preprocess image"
        
        # Make prediction
        prediction = model.predict(processed_image)
        
        # Assuming binary classification (0: Normal, 1: Pneumonia)
        # Adjust this based on your model's output format
        confidence = float(prediction[0][0])
        print(confidence)
        
        if confidence > 0.5:
            result = "Pneumonia Detected"
            probability = confidence
        else:
            result = "Normal (No Pneumonia)"
            probability = 1 - confidence
        
        return result, f"Confidence: {probability:.2%}"
    
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, f"Prediction error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result, confidence = predict_pneumonia(filepath)
        
        if result is None:
            flash(f'Error processing image: {confidence}')
            return redirect(url_for('index'))
        
        # Convert image to base64 for display
        with open(filepath, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return render_template('result.html', 
                             result=result, 
                             confidence=confidence,
                             image_data=img_data)
    
    flash('Invalid file type. Please upload an image file.')
    return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result, confidence = predict_pneumonia(filepath)
        
        # Clean up
        os.remove(filepath)
        
        if result is None:
            return jsonify({'error': confidence}), 500
        
        return jsonify({
            'result': result,
            'confidence': confidence,
            'status': 'success'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    if model is None:
        print("Warning: Model could not be loaded. Please check the model file.")
    app.run(debug=True, host='0.0.0.0', port=8080)
