from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from utils.file_utils import predict_image

app = Flask(__name__)
CORS(app)

# Upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "AI Image Forensics Backend is running"
    })

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file found"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # 🔥 AI vs Real prediction
    analysis = predict_image(file_path)

    return jsonify({
        "message": "Analysis complete",
        "filename": file.filename,
        "analysis": analysis
    })

if __name__ == "__main__":
    app.run(debug=True)
