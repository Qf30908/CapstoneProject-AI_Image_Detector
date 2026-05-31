import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from predictor import load_model, predict_image

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"])

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once when server starts
model = load_model()


@app.route("/")
def home():
    return jsonify({
        "message": "AI Detector API Running"
    })


# =========================
# PREDICT ENDPOINT
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "Empty filename"
        }), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result = predict_image(model, filepath)

    return jsonify(result)


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True)