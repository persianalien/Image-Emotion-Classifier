from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('D:/software engineering/image analysis project/sentiment.h5')

@app.route('/')
def home():
    return render_template("app.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.content_type.startswith('image/'):
        return jsonify({'error': 'Uploaded file is not an image'}), 400
    
    try:
        # Preprocess Uploaded Image
        image = Image.open(file).resize((128, 128))  
        image = np.expand_dims(np.array(image) / 255.0, axis=0)  

        # Predict by loaded model
        pred = model.predict(image)
        class_idx = np.argmax(pred, axis=1)[0]  

        # Map class index to emotion label
        classes = ['angry','happy','sad']
        predicted_label = classes[class_idx]

        return jsonify({'class': predicted_label, 'confidence': float(np.max(pred))})
    
    except Exception as e:
        return jsonify({'error': f'Error processing image: {e}'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
