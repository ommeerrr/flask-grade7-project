from flask import Flask, render_template, request, flash, redirect, url_for
import pickle
import numpy as np
import json
import random

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for flashing messages

# Load the trained model and vectorizer
with open("mc_classifier.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load the dataset for dynamic quiz generation
with open("dataset.txt", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Filter multiple-choice questions
mc_questions = [entry for entry in dataset if entry["type"] == "Multiple Choice"]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        instruction = request.form.get("instruction", "").strip()
        input_text = request.form.get("input_text", "").strip()
        options = [
            request.form.get("option1", "").strip(),
            request.form.get("option2", "").strip(),
            request.form.get("option3", "").strip(),
            request.form.get("option4", "").strip()
        ]

        # Validate input
        if not instruction or not input_text or not all(options):
            flash("All fields are required!", "error")
            return redirect(url_for("index"))

        # Combine question and options for prediction
        question_text = instruction + " " + input_text
        predictions = []
        for option in options:
            combined_text = question_text + " " + option
            vectorized_text = vectorizer.transform([combined_text])
            prediction = model.predict_proba(vectorized_text)[0][1]  # Probability of being correct
            predictions.append(prediction)

        # Find the option with the highest probability
        correct_option_index = np.argmax(predictions)
        correct_option = options[correct_option_index]

        # Render the result page
        return render_template("result.html", correct_option=correct_option)

    # Render the input form for GET requests
    return render_template("index.html")

@app.route("/random", methods=["GET"])
def random_quiz():
    # Select a random question
    if not mc_questions:
        flash("No questions available in the dataset!", "error")
        return redirect(url_for("index"))

    question = random.choice(mc_questions)
    return render_template("index.html", question=question)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
