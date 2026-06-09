from pathlib import Path
import re

import joblib
from flask import Flask, render_template, request


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "svm_url_spam_classifier.joblib"
model = joblib.load(MODEL_PATH)


def preprocess_url(url):
    url = url.lower()
    tokens = re.split(r"[^a-zA-Z0-9]+", url)
    tokens = [token for token in tokens if token]
    return " ".join(tokens)


def predict_url(url):
    processed_url = preprocess_url(url)
    prediction = model.predict([processed_url])[0]
    return "spam" if prediction else "no spam"


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    url = ""

    if request.method == "POST":
        url = request.form.get("url", "").strip()

        if url:
            prediction = predict_url(url)

    return render_template("index.html", prediction=prediction, url=url)


if __name__ == "__main__":
    app.run(debug=True)
