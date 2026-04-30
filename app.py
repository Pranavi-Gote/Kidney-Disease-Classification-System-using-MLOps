from flask import Flask, request, jsonify, render_template, send_file
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline
from cnnClassifier.utils.pdf_report import generate_pdf


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')




@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    # os.system("dvc repro")
    return "Training done successfully!"



@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)

    result = clApp.classifier.predict()

    print("DEBUG RESULT:", result)

    label = result[0]["image"]
    confidence = result[0]["confidence"]

    # Generate PDF automatically
    pdf_filename = os.path.basename(generate_pdf(label, confidence))
    pdf_path = f"/reports/{pdf_filename}"

    if label == "Tumor":
        treatment = [
            "Immediate specialist consultation recommended",
            "Consult Nephrologist / Oncologist",
            "Further CT / MRI diagnosis",
            "Blood and Urine diagnostic tests",
            "Biopsy if recommended by doctor"
        ]

        doctors = [
            "Nephrologist",
            "Urologist",
            "Oncologist",
            "Radiologist"
        ]

    else:
        treatment = [
            "No major abnormality detected",
            "Maintain proper hydration",
            "Healthy balanced diet",
            "Regular exercise",
            "Periodic health check-up recommended"
        ]

        doctors = [
            "General Physician",
            "Nephrologist (for regular checkup)"
        ]


    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "report_path": pdf_path,
        "treatment": treatment,
        "doctors": doctors
    })


@app.route("/reports/<filename>")
def download_report(filename):
    path = os.path.join("reports", filename)
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    clApp = ClientApp()

    app.run(host='0.0.0.0', port=8080) #for AWS

