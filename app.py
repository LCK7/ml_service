from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Cargar modelo y vectorizador
model = joblib.load("complexity_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

class WordRequest(BaseModel):
    word: str

def extract_features(word):
    word = str(word)

    # TF-IDF
    tfidf_features = vectorizer.transform([word]).toarray()

    # Features manuales
    word_length = len(word)
    num_vowels = sum(1 for c in word.lower() if c in "aeiouáéíóú")
    num_consonants = sum(
        1 for c in word.lower()
        if c.isalpha() and c not in "aeiouáéíóú"
    )
    vowel_ratio = num_vowels / word_length if word_length > 0 else 0

    manual_features = np.array([
        [word_length, num_vowels, num_consonants, vowel_ratio, 0]
    ])

    return np.hstack((tfidf_features, manual_features))

@app.post("/predict")
def predict(request: WordRequest):
    features = extract_features(request.word)
    prediction = model.predict(features)[0]

    return {
        "word": request.word,
        "complexity": float(prediction)
    }