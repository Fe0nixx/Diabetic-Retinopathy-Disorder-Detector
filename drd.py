from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model_drd.h5')

# Preprocess image function
def preprocess_image(img):
    img = img.resize((224, 224))  # Adjust size to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array / 255.0) * 2 - 1  # Normalize between -1 and 1
    return img_array

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('image')
        if not data:
            return jsonify({'error': 'No image data received'}), 400

        # Decode and preprocess the image
        image_data = base64.b64decode(data)
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        preprocessed_img = preprocess_image(img)

        # Make a prediction
        prediction = model.predict(preprocessed_img)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        return jsonify({'predicted_class': predicted_class})

    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': 'Failed to process image.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
