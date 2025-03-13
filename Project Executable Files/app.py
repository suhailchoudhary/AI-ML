import os
import numpy as np
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model_path='./src/eye_vgg19.h5'
model = load_model(model_path)

# Define class names (adjust as needed)
class_names = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]

def prepare_image(img_path):
    """Load and preprocess the image."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route("/",methods=["GET", "POST"])
def index():
    return render_template("index.html",)
@app.route('/result', methods=["GET", "POST"])
def result1():
    result = None
    filename = None

    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            result = {"error": "No file selected."}
        else:
            file = request.files["file"]
            filename = file.filename
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Preprocess and predict
            img_array = prepare_image(file_path)
            preds = model.predict(img_array)
            pred_idx = np.argmax(preds, axis=1)[0]
            pred_class = class_names[pred_idx]

            # Map class names to formatted disease names
            disease_mapping = {
                "cataract": "Cataract",
                "diabetic_retinopathy": "Diabetic Retinopathy",
                "glaucoma": "Glaucoma",
                "normal": "Normal"
            }

            formatted_class = disease_mapping.get(pred_class, pred_class)

            confidence = np.max(preds)

            result = {"prediction": formatted_class, "confidence": float(confidence)}
    return render_template("result1.html",  result=result, filename=filename)


if __name__ == "__main__":
    app.run(debug=True)
