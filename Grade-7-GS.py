import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset
with open("dataset.txt", "r", encoding="utf-8") as file:
    data = json.load(file)

# Separate MC questions and validate options
mc_data = []
invalid_entries = []

for entry in data:
    if entry["type"] == "Multiple Choice":
        normalized_options = [opt.strip().replace("  ", " ") for opt in entry["options"]]
        normalized_correct = entry["correct_option"].strip().replace("  ", " ")
        
        if normalized_correct not in normalized_options:
            invalid_entries.append(entry["id"])
        else:
            # Include options in the input text
            entry_text = (
                entry["instruction"] + " " +
                entry["input"] + " " +
                " | ".join(entry["options"])  # Explicitly include options
            )
            mc_data.append({
                "text": entry_text,
                "options": normalized_options,
                "correct_option": normalized_correct
            })

if invalid_entries:
    raise ValueError(f"Correct option mismatch in: {invalid_entries}")

# Prepare features and labels
X = []
y = []

for entry in mc_data:
    question_text = entry["text"]
    options = entry["options"]
    correct_option = entry["correct_option"]
    
    # Create a separate training example for each option
    for option in options:
        X.append(question_text + " " + option)  # Combine question + option
        y.append(1 if option == correct_option else 0)  # Label: 1 for correct, 0 for incorrect

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorize text (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Fixed Model Accuracy: {accuracy:.4f}")

# Save models
with open("mc_classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)