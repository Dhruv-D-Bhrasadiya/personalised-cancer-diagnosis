import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import joblib

# Load model and vectorizers
model = joblib.load("logreg_tfidf_model.joblib")
tfidf_text = joblib.load("tfidf_text_vectorizer.joblib")
tfidf_gene = joblib.load("tfidf_gene_vectorizer.joblib")
tfidf_var = joblib.load("tfidf_var_vectorizer.joblib")

# Preprocessing function (same as training)
stop_words = set(stopwords.words('english'))
def nlp_preprocessing(text):
    if type(text) is not int:
        text = re.sub('[^a-zA-Z0-9\n]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        return ' '.join([word for word in text.split() if word not in stop_words])
    else:
        return ""

# --- User Input Section ---
gene = input("Enter Gene: ").strip()
var = input("Enter Variation: ").strip()
text = input("Enter Text: ").strip()
actual_class = input("Enter Actual Class (or leave blank if unknown): ").strip()
actual_class = actual_class if actual_class else None

# Preprocess
gene_proc = gene
var_proc = var
text_proc = nlp_preprocessing(text)

# Transform features
gene_feat = tfidf_gene.transform([gene_proc])
var_feat = tfidf_var.transform([var_proc])
text_feat = tfidf_text.transform([text_proc])
X = np.hstack([gene_feat.toarray(), var_feat.toarray(), text_feat.toarray()])

# Predict
pred_class = model.predict(X)[0]

# Feature presence
gene_vocab = tfidf_gene.get_feature_names_out()
var_vocab = tfidf_var.get_feature_names_out()
text_vocab = tfidf_text.get_feature_names_out()

present_genes = [g for g in gene_vocab if g in gene_proc.lower()]
present_vars = [v for v in var_vocab if v in var_proc.lower()]
present_words = [w for w in text_vocab if w in text_proc.split()]

print("\n--- Prediction Result ---")
print(f"Gene: {gene} | Present in vocab: {present_genes if present_genes else 'Absent'}")
print(f"Variation: {var} | Present in vocab: {present_vars if present_vars else 'Absent'}")
print(f"Text (first 100 chars): {text[:100]}...")
print(f"Words present in vocab: {present_words[:10]}{'...' if len(present_words)>10 else ''}")
print(f"Predicted Class: {pred_class}")
if actual_class:
    print(f"Actual Class: {actual_class}")

# Show top contributing features (optional, for interpretability)
# Show top contributing features (optional, for interpretability)
coefs = model.estimator.coef_[pred_class-1]  # pred_class-1 because classes are 1-indexed
feature_names = np.concatenate([gene_vocab, var_vocab, text_vocab])
top_indices = np.argsort(coefs)[-10:][::-1]
print("\nTop features contributing to this prediction:")
for idx in top_indices:
    print(f"{feature_names[idx]}: {coefs[idx]:.4f}")