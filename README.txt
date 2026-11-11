AI PHISHING EMAIL DETECTOR - WINDOWS SETUP

1. Open this folder in Visual Studio Code.
2. Open Terminal > New Terminal.
3. Run:
   py -m venv .venv
   .venv\Scripts\activate
   py -m pip install --upgrade pip
   py -m pip install -r requirements.txt
4. Train the model:
   py phishing_detector.py
5. Start the website:
   py app.py
6. Open http://127.0.0.1:5000 in your browser.

Keep app.py, phishing_detector.py, requirements.txt and the CSV in this same folder.
Keep templates/index.html inside the templates folder.
The training script creates phishing_detector.pkl, model_comparison.csv and confusion_matrix.png.
If you change the dataset or training code, run phishing_detector.py again before app.py.
