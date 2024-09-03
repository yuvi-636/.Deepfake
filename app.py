import os
from flask import Flask, request, render_template, redirect
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Paths
MODEL_PATH = 'keras_Model.h5'  # Path to the trained model
LABELS_PATH = 'labels.txt'     # Path to the labels file
UPLOAD_FOLDER = 'uploads'      # Folder to store uploaded images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the model
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH, compile=False)
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load class names
class_names = open(LABELS_PATH, "r").readlines()

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(image_path):
    """Prepare the image for prediction."""
    size = (224, 224)  # Required size by the model
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

@app.route('/')
def index():
    """Render the upload form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and make prediction."""
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Prepare and predict the image
        prepared_image = prepare_image(file_path)
        prediction = model.predict(prepared_image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Return the prediction result
        result = f"Class: {class_name[2:]}, Confidence Score: {confidence_score:.2f}"
        return render_template('result.html', result=result)
    return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=True)
