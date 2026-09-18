from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import os

MODEL_PATH = "my_vgg_model.h5"
CLASS_LABELS = ["glioma", "meningioma", "notumor", "pituitary"]

app = Flask(__name__)
CORS(app)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH}' was not found. "
        "Place the trained model in the project directory before starting the API."
    )

model = load_model(MODEL_PATH)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files["file"]

    try:
        img = image.load_img(
            io.BytesIO(uploaded_file.read()),
            target_size=(124, 124)
        )
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0

        predictions = model.predict(x, verbose=0)[0]
        predicted_idx = int(np.argmax(predictions))

        return jsonify({
            "predicted_class": CLASS_LABELS[predicted_idx],
            "confidence": round(float(predictions[predicted_idx]), 4),
            "probabilities": {
                label: round(float(predictions[i]), 4)
                for i, label in enumerate(CLASS_LABELS)
            }
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
