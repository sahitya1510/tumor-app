# backend/app.py
import io
import os
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from report_generator import build_report_and_facts

app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    # optional patient meta from UI
    patient = {
        "name": request.form.get("name"),
        "age": request.form.get("age", type=int),
        "sex": request.form.get("gender"),
    }

    file = request.files.get("image") or request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No image uploaded (field must be 'image' or 'file')"}), 400

    image_bytes = file.read()

    try:
        pdf_bytes, facts, _overlay_png = build_report_and_facts(image_bytes, patient=patient)
    except Exception as e:
        # make errors obvious in frontend
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="Brain_Tumor_Report.pdf",
    )

if __name__ == "__main__":
    # Run local dev server
    app.run(host="127.0.0.1", port=5000, debug=True)
