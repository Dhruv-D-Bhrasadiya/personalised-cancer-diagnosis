import streamlit as st
import joblib
import re
import numpy as np
from scipy.sparse import hstack
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords

# Load model and vectorizers
@st.cache_resource
def load_model_and_vectorizers():
    model = joblib.load("final_lr_model.pkl")
    gene_vectorizer = joblib.load("gene_vectorizer.pkl")
    variation_vectorizer = joblib.load("variation_vectorizer.pkl")
    text_vectorizer = joblib.load("text_vectorizer.pkl")
    return model, gene_vectorizer, variation_vectorizer, text_vectorizer

model, gene_vectorizer, variation_vectorizer, text_vectorizer = load_model_and_vectorizers()

stop_words = set(stopwords.words('english'))

def nlp_preprocessing(total_text):
    if type(total_text) is not int:
        string = ""
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', total_text)
        total_text = re.sub(r'\s+', ' ', total_text)
        total_text = total_text.lower()
        for word in total_text.split():
            if word not in stop_words:
                string += word + " "
        return string.strip()
    return ""

def predict_and_interpret(gene, variation):
    gene_proc = gene.lower()
    variation_proc = variation.lower()
    # For interpretability, combine gene and variation as text
    text = nlp_preprocessing(f"{gene} {variation}")

    gene_feat = gene_vectorizer.transform([gene_proc])
    var_feat = variation_vectorizer.transform([variation_proc])
    text_feat = text_vectorizer.transform([text])
    text_feat = normalize(text_feat, axis=0)
    x = hstack([gene_feat, var_feat, text_feat]).tocsr()

    pred_class = model.predict(x)[0]
    pred_proba = model.predict_proba(x)[0]

    # Interpretability
    gene_features = gene_vectorizer.get_feature_names_out()
    variation_features = variation_vectorizer.get_feature_names_out()
    text_features = text_vectorizer.get_feature_names_out()
    text_words = set(text.split())

    coefs = model.estimator.coef_[pred_class-1]
    n_gene = len(gene_features)
    n_var = len(variation_features)
    n_text = len(text_features)

    # Show all features sorted by absolute importance
    indices = np.argsort(-np.abs(coefs))
    feature_table = []
    for idx in indices:
        if idx < n_gene:
            present = (gene_features[idx] == gene_proc)
            feature_table.append(("Gene", gene_features[idx], present, coefs[idx]))
        elif idx < n_gene + n_var:
            v_idx = idx - n_gene
            present = (variation_features[v_idx] == variation_proc)
            feature_table.append(("Variation", variation_features[v_idx], present, coefs[idx]))
        else:
            t_idx = idx - n_gene - n_var
            if t_idx < n_text:
                word = text_features[t_idx]
                present = word in text_words
                feature_table.append(("Text", word, present, coefs[idx]))

    return pred_class, pred_proba, feature_table

# --- Streamlit UI ---

st.set_page_config(page_title="Personalized Cancer Mutation Classifier", page_icon="🧬", layout="centered")

st.markdown("""
    <style>
    .main {background-color: #f7fafd;}
    .stButton>button {background-color: #1976d2; color: white;}
    .stTextInput>div>input {background-color: #e3f2fd;}
    .stTable {background-color: #fff;}
    </style>
""", unsafe_allow_html=True)

st.title("🧬 Personalized Cancer Mutation Classifier")
st.markdown("""
Welcome, Doctor!  
Enter the **Gene** and **Variation** to predict the cancer mutation class.  
The model will also show you the most important features (gene, variation, and text) that led to this prediction, so you can interpret and trust the result.
""")

with st.form("mutation_form"):
    gene = st.text_input("Gene Name", help="e.g. BRCA1, TP53, EGFR")
    variation = st.text_input("Variation Name", help="e.g. Truncating Mutations, W802*")
    submitted = st.form_submit_button("Predict Class")

if submitted:
    # Basic validation: gene/variation should be alphabetic or alphanumeric (no spaces, not just names)
    if not gene or not variation:
        st.warning("Please enter both Gene and Variation.")
    elif not re.match(r"^[A-Za-z0-9_\-\*]+$", gene.strip()) or not re.match(r"^[A-Za-z0-9_\-\* ]+$", variation.strip()):
        st.error("Please enter a valid Gene and Variation name (e.g., BRCA1, TP53, Truncating Mutations, W802*).")
    else:
        with st.spinner("Predicting and interpreting..."):
            pred_class, pred_proba, feature_table = predict_and_interpret(gene, variation)
        st.success(f"Predicted Class: **{pred_class}**")
        st.markdown("#### Class Probabilities")
        st.bar_chart(pred_proba)

        st.markdown("#### Important Features for this Prediction")
        st.write("**Present in input?**: ✅ = Yes, ❌ = No")
        st.table([
            {
                "Feature Type": ftype,
                "Feature Name": fname,
                "Present in Input?": "✅" if present else "❌",
                "Importance": f"{coef:.3f}"
            }
            for ftype, fname, present, coef in feature_table[:20]
        ])

        st.markdown("""
        <small>
        <i>
        This tool is for clinical decision support. Please review the important features and use your medical expertise to interpret the results.
        </i>
        </small>
        """, unsafe_allow_html=True)