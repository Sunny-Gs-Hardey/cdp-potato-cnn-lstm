import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Paths
MODEL_PATH = "cnn_lstm_potato_model.h5"
CLASS_INDICES_PATH = "class_indices.json"
UPLOAD_FOLDER = os.path.join("static", "uploads")

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model & class indices
model = load_model(MODEL_PATH)
with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse mapping (index -> class name)
idx_to_class = {v: k for k, v in class_indices.items()}


# ---------------- ROUTES ---------------- #

@app.route("/")
def welcome():
    """Welcome page"""
    return render_template("welcome.html")


@app.route("/main")
def main_page():
    """Main upload/predict page"""
    return render_template("main.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload & prediction"""
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]

    if file.filename == "":
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Preprocess image
        img = load_img(filepath, target_size=(128, 128))  # adjust if your model uses another size
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        preds = model.predict(img_array)
        pred_idx = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds)) * 100
        label = idx_to_class[pred_idx]

        return render_template(
            "main.html",
            prediction=label,
            confidence=round(confidence, 2),
            img_path=filepath
        )


# Run app
if __name__ == "__main__":
    app.run(debug=True)
