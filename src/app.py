from pathlib import Path
import re

import joblib
from flask import Flask, render_template, request


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "svm_url_spam_classifier.joblib"

try:
    model = joblib.load(MODEL_PATH)
except Exception as error:
    model = None
    app.logger.exception("No se pudo cargar el modelo: %s", error)


def preprocess_url(url):
    url = url.lower()
    tokens = re.split(r"[^a-zA-Z0-9]+", url)
    tokens = [token for token in tokens if token]
    return " ".join(tokens)


def predict_url(url):
    if model is None:
        raise RuntimeError("El modelo no está disponible")

    processed_url = preprocess_url(url)
    prediction = model.predict([processed_url])[0]
    return "spam" if prediction else "no spam"


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error_message = None
    url = ""

    if request.method == "POST":
        url = request.form.get("url", "").strip()

        if url:
            try:
                prediction = predict_url(url)
            except Exception as error:
                app.logger.exception("Error al analizar la URL: %s", error)
                error_message = "No se pudo analizar la URL en este momento. Intentá de nuevo más tarde."

    return render_template(
        "index.html",
        prediction=prediction,
        error_message=error_message,
        url=url,
    )


if __name__ == "__main__":
    app.run(debug=True)
