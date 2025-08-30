from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import json

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model (without compiling to avoid warning)
model = load_model("cnn_lstm_potato_model.h5", compile=False)

# Load class indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}

# Treatment advice
treatment_advice = {
    "Early_Blight": "Use fungicides like chlorothalonil or copper-based sprays. Remove infected leaves and practice crop rotation.",
    "Late_Blight": "Apply fungicides such as mancozeb or metalaxyl. Destroy infected plants and avoid overhead irrigation.",
    "Healthy": "No treatment needed. Maintain good hygiene and monitor regularly."
}

@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/main")
def main_page():
    return render_template("main.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Secure the filename
    filename = file.filename.replace(" ", "_")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess image
    img = load_img(filepath, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)
    predicted_index = int(np.argmax(pred))
    prediction = class_labels.get(predicted_index, "Unknown")
    confidence = round(float(np.max(pred)) * 100, 2)
    formatted_prediction = prediction.replace("_", " ").title()

    treatment = treatment_advice.get(prediction, "No advice available.")

    return jsonify({
        "prediction": formatted_prediction,
        "confidence": confidence,
        "treatment": treatment,
        "image_path": filepath
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
