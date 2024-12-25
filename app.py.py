from flask import Flask, request, render_template, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import torch.nn.functional as F  

app = Flask(__name__, template_folder='templates')

saved_model_dir = 'Model'

def load_models(load_dir):
    sentiment_model_path = os.path.join(load_dir, 'sentiment_model')
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)
    
    tokenizer_path = os.path.join(load_dir, 'tokenizer')
    if os.path.exists(tokenizer_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except ValueError as e:
            print(f"Error loading tokenizer from local path: {e}")
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            tokenizer.save_pretrained(tokenizer_path)
    else:
        print("Tokenizer folder not found. Using a pretrained tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.save_pretrained(tokenizer_path)
    
    vectorizer_path = os.path.join(load_dir, 'vectorizer.joblib')
    if os.path.exists(vectorizer_path):
        try:
            vectorizer = joblib.load(vectorizer_path)
            check_is_fitted(vectorizer, attributes=["idf_"])
        except (NotFittedError, FileNotFoundError, ValueError):
            print("Vectorizer not fitted or file corrupted. Retraining vectorizer.")
            vectorizer = retrain_vectorizer()
    else:
        print("Vectorizer not found. Retraining vectorizer.")
        vectorizer = retrain_vectorizer()

    classifier_categories = joblib.load(os.path.join(load_dir, 'classifier_categories.joblib'))
    classifier_activities = joblib.load(os.path.join(load_dir, 'classifier_activities.joblib'))
    mlb_categories = joblib.load(os.path.join(load_dir, 'mlb_categories.joblib'))
    mlb_activities = joblib.load(os.path.join(load_dir, 'mlb_activities.joblib'))
    
    return sentiment_model, tokenizer, vectorizer, classifier_categories, classifier_activities, mlb_categories, mlb_activities

def retrain_vectorizer():
    sample_texts = [
        "This is a great place to visit!",
        "I loved the food and atmosphere.",
        "The service was bad, but the location is amazing."
    ]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(sample_texts)
    vectorizer_path = os.path.join(saved_model_dir, 'vectorizer.joblib')
    joblib.dump(vectorizer, vectorizer_path)
    print("Vectorizer retrained and saved.")
    return vectorizer

sentiment_model, tokenizer, vectorizer, classifier_categories, classifier_activities, mlb_categories, mlb_activities = load_models(saved_model_dir)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("review", "").strip()

        if not text:
            return render_template("index.html", result="No text provided", sentiment="", sentiment_score=None, categories=[], activities=[])

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        inputs.pop('token_type_ids', None)
        
        with torch.no_grad():
            sentiment_output = sentiment_model(**inputs)
            logits = sentiment_output.logits
            sentiment_pred = torch.argmax(logits, dim=-1).item()

            sentiment_prob = F.softmax(logits, dim=-1).max().item()

        text_vector = vectorizer.transform([text])
        categories_pred = classifier_categories.predict(text_vector)
        activities_pred = classifier_activities.predict(text_vector)

        categories_labels = mlb_categories.inverse_transform(categories_pred)
        activities_labels = mlb_activities.inverse_transform(activities_pred)

        sentiment = "positive" if sentiment_pred == 1 else "negative"
        return render_template("index.html", result="Prediction Successful", sentiment=sentiment, 
                               sentiment_score=sentiment_prob, categories=categories_labels, activities=activities_labels)
    
    return render_template("index.html", result="", sentiment="", sentiment_score="", categories=[], activities=[])

if __name__ == "__main__":
    app.run(debug=True)
