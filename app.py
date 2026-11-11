"""Flask web application for the phishing email detector."""
import re
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, render_template, request
from scipy.sparse import csr_matrix, hstack

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "phishing_detector.pkl"

SUSPICIOUS_WORDS = [
    "urgent", "verify", "password", "account", "suspended", "click",
    "bank", "login", "security", "confirm", "payment", "update",
    "winner", "free", "limited", "immediately", "expire", "claim",
    "credential", "invoice", "gift", "prize", "reset", "locked"
]


def preprocess_email(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"[^a-z0-9\s@]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_features(text):
    text = str(text)
    lower = text.lower()
    return [
        len(text), len(text.split()), text.count("!"), text.count("@"),
        text.count("."), text.count("/"),
        len(re.findall(r"https?://\S+|www\.\S+", text, re.I)),
        sum(char.isupper() for char in text),
        sum(lower.count(word) for word in SUSPICIOUS_WORDS),
    ]


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


model_data = load_model()
history = []


def analyze_email(email):
    if model_data is None:
        raise RuntimeError("AI model is missing. Train it first using: python phishing_detector.py")

    vectorizer = model_data["vectorizer"]
    scaler = model_data["scaler"]
    model = model_data["model"]

    text_features = vectorizer.transform([email])
    numeric = np.asarray([extract_features(email)], dtype=float)
    numeric_features = scaler.transform(csr_matrix(numeric))
    final_features = hstack([text_features, numeric_features]).tocsr()

    prediction = int(model.predict(final_features)[0])
    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(final_features)[0][1]) * 100
        score_label = "estimated phishing probability"
    else:
        decision = float(model.decision_function(final_features)[0])
        score = float(100 / (1 + np.exp(-np.clip(decision, -30, 30))))
        score_label = "model risk indicator (not a calibrated probability)"

    found_words = [word for word in SUSPICIOUS_WORDS if word in email.lower()]
    urls = re.findall(r"https?://\S+|www\.\S+", email, re.IGNORECASE)
    return {
        "prediction": prediction,
        "score": round(score, 1),
        "score_label": score_label,
        "words": found_words,
        "urls": urls,
        "length": len(email),
        "exclamations": email.count("!"),
    }


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    email = ""
    error = ""

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        if not email:
            error = "Please paste an email before analyzing."
        elif len(email) > 20000:
            error = "The email is too long. Maximum length is 20,000 characters."
        elif model_data is None:
            error = "The trained model was not found. Run python phishing_detector.py first."
        else:
            try:
                result = analyze_email(email)
                history.insert(0, {
                    "prediction": result["prediction"],
                    "preview": email[:90],
                    "score": result["score"],
                })
                del history[10:]
            except Exception as exc:
                app.logger.exception("Email analysis failed")
                error = f"Analysis failed: {exc}"

    return render_template(
        "index.html",
        result=result,
        email=email,
        error=error,
        history=history,
        model_name=model_data.get("model_name", "Machine Learning") if model_data else "Not trained",
    )


if __name__ == "__main__":
    if model_data is None:
        print("WARNING: Model not found. First run: python phishing_detector.py")
    app.run(host="127.0.0.1", port=5000, debug=True)
