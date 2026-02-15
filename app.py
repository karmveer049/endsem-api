from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)


# load weights once
with open("weights.json","r") as f:
    W = json.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    Attendance = data["attendance"]
    MidSemMarks = data["midsem"]
    IQLevel = data["iq"]
    SelfStudy = data["study"]
    Attentiveness = data["attentive"]

    e=2.71828
    c1,c2,c3,c4,c5=2,3,4,5,6
    sigma=100

    X_input=[
       e**(-((Attendance)-c1)**2/(2*sigma**2)),
       e**(-((MidSemMarks)-c2)**2/(2*sigma**2)),
       e**(-((IQLevel)-c3)**2/(2*sigma**2)),
       e**(-((SelfStudy)-c4)**2/(2*sigma**2)),
       e**(-((Attentiveness)-c5)**2/(2*sigma**2))
    ]

    pred = sum(W[i][0] * X_input[i] for i in range(len(W)))
    return jsonify({"prediction": pred})

if __name__ == "__main__":
    app.run()
