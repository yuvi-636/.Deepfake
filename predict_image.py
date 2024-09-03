from keras.models import load_model  
from PIL import Image, ImageOps 
import numpy as np
import sys

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

def load_labels(labels_path):
    """Load and process the labels file."""
    with open(labels_path, "r") as file:
        class_names = [line.strip() for line in file.readlines()]
    return class_names

def prepare_image(image_path):
    """Prepare the image for prediction."""
    size = (224, 224)  # Required size by the model
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    return data

def predict_image(model, image_data, class_names):
    """Predict the class of the image using the model."""
    prediction = model.predict(image_data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

def main(image_path, model_path="keras_Model.h5", labels_path="labels.txt"):
    """Main function to handle the prediction process."""
    # Load the model
    try:
        model = load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Load the labels
    class_names = load_labels(labels_path)

    # Prepare the image
    image_data = prepare_image(image_path)

    # Predict the image
    class_name, confidence_score = predict_image(model, image_data, class_names)

    # Print the prediction and confidence score
    print(f"Class: {class_name}, Confidence Score: {confidence_score:.2f}")

if __name__ == "__main__":
    # Replace '<IMAGE_PATH>' with the path to your image file
    image_path = '<IMAGE_PATH>'
    main(image_path)
