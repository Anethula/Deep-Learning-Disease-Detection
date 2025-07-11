# Pneumonia Detection Web App

A Flask-based web application that uses a VGG19 deep learning model to detect pneumonia in chest X-ray images.

## Features

- **AI-Powered Analysis**: Uses VGG19 convolutional neural network for medical image analysis
- **User-Friendly Interface**: Modern, responsive web interface with drag-and-drop functionality
- **Real-time Results**: Get instant analysis results with confidence scores
- **Secure Processing**: Images are processed locally and automatically deleted after analysis
- **API Endpoint**: RESTful API for programmatic access

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure your VGG19 model file is in the project directory**:
   - The app expects a file named `vgg19_model_03.h5` in the root directory
   - Make sure this is your trained pneumonia detection model

## Usage

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and go to:
   ```
   http://localhost:5000
   ```

3. **Upload a chest X-ray image**:
   - Click "Choose File" or drag and drop an image
   - Supported formats: PNG, JPG, JPEG, GIF, BMP, TIFF
   - Maximum file size: 16MB

4. **Get results**:
   - The AI will analyze the image and provide a diagnosis
   - Results include confidence scores and visual feedback

## API Usage

You can also use the API endpoint for programmatic access:

```bash
curl -X POST -F "file=@your_xray_image.jpg" http://localhost:5000/api/predict
```

Response format:
```json
{
  "result": "Normal (No Pneumonia)",
  "confidence": "Confidence: 85.32%",
  "status": "success"
}
```

## File Structure

```
NN Disease Detection/
├── app.py                 # Main Flask application
├── vgg19_model_03.h5     # Your trained VGG19 model
├── requirements.txt      # Python dependencies
├── templates/
│   ├── index.html        # Main upload page
│   └── result.html       # Results display page
└── uploads/              # Temporary upload directory (auto-created)
```

## Model Information

- **Architecture**: VGG19 Convolutional Neural Network
- **Input Size**: 224x224 RGB images
- **Output**: Binary classification (Normal/Pneumonia)
- **Preprocessing**: Images are automatically resized and normalized

## Important Notes

⚠️ **Medical Disclaimer**: This application is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis, treatment, or advice. Always consult with qualified healthcare professionals for medical concerns.

## Troubleshooting

**Model Loading Issues**:
- Ensure `vgg19_model_03.h5` is in the correct directory
- Check that the model file is not corrupted
- Verify TensorFlow compatibility

**OpenCV Issues**:
- If you encounter OpenCV import errors, try:
  ```bash
  pip uninstall opencv-python
  pip install opencv-python-headless
  ```

**Memory Issues**:
- Large images may require significant memory
- Consider reducing image sizes if you encounter memory errors

## Development

To run in development mode:
```bash
export FLASK_ENV=development
python app.py
```

The application will run on `http://0.0.0.0:5000` by default, making it accessible from other devices on your network.
