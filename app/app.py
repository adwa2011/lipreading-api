from flask import Flask, request, jsonify
import os
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model
import numpy as np
import imageio

app = Flask(__name__)

# Set the data directory
data_s1_path = os.path.join(os.path.expanduser(
    "~"), "Desktop", "Lip-Reader", "data", "s1")


@app.route('/predict', methods=['POST'])
def predict():
    # Check if a video file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Save the uploaded file
    file_path = os.path.join(data_s1_path, 'bbaf2n.mpg')
    file.save(file_path)

    # Load the saved video
    video, _ = load_data(tf.convert_to_tensor(file_path))

    # Load the model
    model = load_model()

    # Predict lip reading
    yhat = model.predict(tf.expand_dims(video, axis=0))
    decoder = tf.keras.backend.ctc_decode(
        yhat, [75], greedy=True)[0][0].numpy()

    # Convert prediction to text
    converted_prediction = tf.strings.reduce_join(
        num_to_char(decoder)).numpy().decode('utf-8')

    return jsonify({'prediction': converted_prediction})


if __name__ == '__main__':
    app.run(debug=True)
