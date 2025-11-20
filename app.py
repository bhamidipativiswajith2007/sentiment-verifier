from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load your trained pipeline
model = joblib.load("sentiment_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    review_text = request.form["review"]

    prediction = model.predict([review_text])[0]

    label = "Positive 😄" if prediction == 1 else "Negative 😞"

    return render_template("index.html", result=label)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
