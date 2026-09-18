import io
import os

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = os.getenv("MODEL_PATH", "brain_tumor_resnet50.keras")
IMAGE_SIZE = (124, 124)
CLASS_LABELS = ["glioma", "meningioma", "notumor", "pituitary"]

app = Flask(__name__)
CORS(app)

model = load_model(MODEL_PATH)


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "message": "Brain Tumor MRI classification API is running."
    })


@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        return jsonify({"error": "No file selected."}), 400

    try:
        img = image.load_img(
            io.BytesIO(uploaded_file.read()),
            target_size=IMAGE_SIZE,
            color_mode="rgb",
        )
        x = image.img_to_array(img).astype("float32") / 255.0
        x = np.expand_dims(x, axis=0)

        predictions = model.predict(x, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))

        return jsonify({
            "predicted_class": CLASS_LABELS[predicted_index],
            "confidence": round(float(predictions[predicted_index]), 4),
            "probabilities": {
                label: round(float(predictions[i]), 4)
                for i, label in enumerate(CLASS_LABELS)
            },
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)
