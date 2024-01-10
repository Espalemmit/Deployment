from flask import Flask, render_template, request, jsonify
import pickle

cv = pickle.load(open("models/cv.pkl", "rb"))
clf = pickle.load(open("models/clf.pkl", "rb"))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    text = ""
    if request.method == "POST":
        text = request.form.get("email-content")
    return render_template("index.html", text = text)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.get('content')
    email = data['content']
    tokenized_email = cv.transform([email])
    prediction = clf.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    return render_template("index.html", prediction=prediction, email=email)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    email = data['content']
    tokenized_email = cv.transform(email)
    prediction = clf.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    return jsonify({'prediction' : prediction, 'email' : email})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)