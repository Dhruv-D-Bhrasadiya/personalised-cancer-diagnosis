import streamlit as st
import numpy as np
import pandas as pd
import re
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Personalised Cancer Diagnosis | AI-Powered Genomics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #0e1117;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #e0e7ff;
        margin: 0.8rem 0 0 0;
        font-size: 1.3rem;
        font-weight: 400;
    }
    
    .info-card {
        background: transparent;
        padding: 2rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        color: white;   
    }
    .info-card h3 {
        font-size: 1.5rem;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    .info-card p {
        font-size: 1.05rem;
        color: white;
        line-height: 1.5;
        margin: 0;
    }
    .info-card strong {
        color: #ffffff;
        font-weight: 600;
    }
    
    .result-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        color: black;
    }
    .prediction-box {
        background: white;
        padding: 2rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        border: 3px solid #667eea;
    }
    .prediction-box h2 {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    .prediction-box h1 {
        font-size: 3.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box p {
        font-size: 1.4rem;
        margin-top: 0.5rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.15rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102,126,234,0.4);
    }
    st.image("heix.png", use_column_width=True, caption="Precision Genomics")
    
    .sidebar-info {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #667eefa;
    }
            
    /* Hide empty containers above footer logos */
    .element-container:has(.footer-logo-container) ~ .element-container:empty {
        display: none !important;
    }

    /* Alternative - hide all empty elements in columns */
    .column .element-container:empty {
        display: none !important;
    }
    
    /* Footer Logo Styling */
    .footer-logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 1.5rem;
        background: white;
        border-radius: 10px;
        height: 160px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 0.5rem;
    }
    .footer-logo-container img {
        max-height: 110px;
        width: auto;
        object-fit: contain;
    }
    
    /* Footer Section */
    .footer-section {
        text-align: center;
        padding: 2rem 1.5rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #e8edf2 100%);
        border-radius: 15px;
        margin-top: 3rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .footer-section h3 {
        color: #667eea;
        font-size: 2rem;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .footer-section p {
        color: #555;
        font-size: 1.15rem;
        margin-bottom: 2rem;
    }
    
    /* Caption Styling */
    .stCaption {
        text-align: center;
        font-size: 1rem;
        color: #666;
        margin-top: 0.8rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize session state ---
if 'example_gene' not in st.session_state:
    st.session_state.example_gene = ""
if 'example_var' not in st.session_state:
    st.session_state.example_var = ""
if 'example_text' not in st.session_state:
    st.session_state.example_text = ""

# --- Load model and vectorizers ---
@st.cache_resource
def load_models():
    model = joblib.load("logreg_tfidf_model.joblib")
    tfidf_text = joblib.load("tfidf_text_vectorizer.joblib")
    tfidf_gene = joblib.load("tfidf_gene_vectorizer.joblib")
    tfidf_var = joblib.load("tfidf_var_vectorizer.joblib")
    return model, tfidf_text, tfidf_gene, tfidf_var

model, tfidf_text, tfidf_gene, tfidf_var = load_models()

# --- Preprocessing function ---
# Hardcoded English stopwords to avoid NLTK download issues
stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', 
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 
    'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 
    'for', 'with', 'through', 'during', 'before', 'after', 'above', 'below', 
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
    'then', 'once'
])

def nlp_preprocessing(text):
    if type(text) is not int:
        text = re.sub('[^a-zA-Z0-9\n]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        return ' '.join([word for word in text.split() if word not in stop_words])
    else:
        return ""

# --- Sidebar ---
with st.sidebar:
    # Try to load MSK logo, fallback if not found
    try:
        st.image("msk-logo.png", width=230)
    except:
        st.markdown("### 🏥 MSK Cancer Center")
    
    st.markdown("### 🏥 About This Project")
    
    st.markdown("""
    <div class="sidebar-info">
    <strong>Developed for:</strong><br>
    Memorial Sloan Kettering Cancer Center (MSK)<br><br>
    <strong>Competition:</strong><br>
    MSK - Redefining Cancer Treatment<br>
    NIPS 2017 Competition Track
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📚 Medical Glossary")
    with st.expander("🧬 What is a Gene?"):
        st.markdown("""
        A **gene** is a segment of DNA that contains instructions for making proteins. 
        Genes determine traits and regulate biological functions.
        
        **Common Cancer-Related Genes:**
        - **BRCA1/BRCA2**: Breast cancer susceptibility
        - **TP53**: Tumor suppressor ("guardian of genome")
        - **EGFR**: Epidermal growth factor receptor
        - **KRAS**: Cell signaling protein
        - **PIK3CA**: Cell growth and division
        """)
    
    with st.expander("🔬 What is a Variation/Mutation?"):
        st.markdown("""
        A **genetic variation** (mutation) is a change in the DNA sequence. 
        
        **Types of Mutations:**
        - **Missense**: One amino acid replaced with another (e.g., A1699S)
        - **Nonsense**: Creates a stop signal
        - **Deletion**: Part of gene removed
        - **Insertion**: Extra DNA added
        - **Frameshift**: Changes reading frame
        
        **Example:** `A1699S` means Alanine at position 1699 changed to Serine
        """)
    
    with st.expander("📊 Classification Classes"):
        st.markdown("""
        The model classifies variants into **9 classes** based on clinical evidence:
        
        - **Class 1-3**: Likely neutral/benign variants
        - **Class 4-6**: Variants of uncertain significance
        - **Class 7-9**: Likely pathogenic/actionable variants
        
        *Higher classes often indicate stronger clinical significance*
        """)
    
    st.markdown("### 🔗 Helpful Resources")
    st.markdown("""
    - [MSK Cancer Center](https://www.mskcc.org/)
    - [National Cancer Institute](https://www.cancer.gov/)
    - [ClinVar Database](https://www.ncbi.nlm.nih.gov/clinvar/)
    - [OncoKB](https://www.oncokb.org/)
    """)
    
    # st.markdown("---")
    # st.markdown("""
    # <div style="text-align:center; font-size:0.85rem; color:#666;">
    # <strong>👨‍💻 Developer</strong><br>
    # <a href="https://www.linkedin.com/in/aditya-singh-2b319b299/" target="_blank">Aditya Singh</a><br>
    # <a href="https://github.com/AdiSinghCodes" target="_blank">GitHub Profile</a>
    # </div>
    # """, unsafe_allow_html=True)

# --- Main Content ---
st.markdown("""
<div class="main-header">
    <h1>🧬 Personalised Cancer Diagnosis</h1>
    <p>AI-Powered Genetic Variant Classification System</p>
</div>
""", unsafe_allow_html=True)

# --- Introduction Section ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3 style="color:#667eea;">🎯 Purpose</h3>
        <p>Predict genetic variant classes to support personalised cancer treatment decisions</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3 style="color:#667eea;">🤖 Technology</h3>
        <p>Machine Learning with TF-IDF vectorization and Logistic Regression</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h3 style="color:#667eea;">👥 For</h3>
        <p>Clinicians, researchers, and genomics students</p>
    </div>
    """, unsafe_allow_html=True)

# --- Instructions ---
with st.expander("📖 How to Use This Tool", expanded=False):
    st.markdown("""
    ### Step-by-Step Guide:
    
    1. **Enter Gene Name** (Required)
       - Example: BRCA1, TP53, EGFR, KRAS
       - Use standard gene symbols (HUGO nomenclature)
    
    2. **Enter Variation/Mutation** (Required)
       - Example: A1699S, R273H, Deletion, Amplification
       - Can be amino acid changes, deletions, or other alterations
    
    3. **Add Clinical Text** (Optional but Recommended)
       - Include relevant information from research papers, clinical notes, or lab reports
       - More context improves prediction accuracy
    
    4. **Click "Classify Variant"** to get results
    
    5. **Review Results** including predicted class, confidence, and contributing features
    
    ⚠️ **Disclaimer**: This tool is for research and educational purposes only. 
    It should NOT be used as the sole basis for clinical diagnosis or treatment decisions.
    """)

st.markdown("---")

# --- Example Cases ---
st.markdown("### 💡 Try Quick Examples")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🧪 Example 1: BRCA1"):
        st.session_state.example_gene = "BRCA1"
        st.session_state.example_var = "A1699S"
        st.session_state.example_text = "Missense mutation in BRCA1 associated with breast cancer susceptibility"
        st.rerun()

with col2:
    if st.button("🧪 Example 2: TP53"):
        st.session_state.example_gene = "TP53"
        st.session_state.example_var = "R273H"
        st.session_state.example_text = "Hotspot mutation in TP53 tumor suppressor gene"
        st.rerun()

with col3:
    if st.button("🧪 Example 3: EGFR"):
        st.session_state.example_gene = "EGFR"
        st.session_state.example_var = "Deletion"
        st.session_state.example_text = "EGFR exon 19 deletion commonly found in lung adenocarcinoma"
        st.rerun()

with col4:
    if st.button("🔄 Clear All"):
        st.session_state.example_gene = ""
        st.session_state.example_var = ""
        st.session_state.example_text = ""
        st.rerun()

st.markdown("---")

# --- Input Section ---
st.markdown("### 📝 Enter Variant Information")

col1, col2 = st.columns(2)

with col1:
    gene = st.text_input(
        "🧬 Gene Name *",
        value=st.session_state.example_gene,
        placeholder="e.g., BRCA1, TP53, EGFR",
        help="Enter the official gene symbol (HUGO nomenclature)"
    )

with col2:
    variation = st.text_input(
        "🔬 Variation / Mutation *",
        value=st.session_state.example_var,
        placeholder="e.g., A1699S, R273H, Deletion",
        help="Enter the specific mutation or variation"
    )

text = st.text_area(
    "📄 Clinical Text / Description (Optional)",
    value=st.session_state.example_text,
    placeholder="Enter any relevant clinical information, research findings, or context...",
    help="Provide additional context from medical literature, case reports, or clinical observations",
    height=150
)

st.markdown("<br>", unsafe_allow_html=True)

# --- Classify Button ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run_button = st.button("🚀 Classify Variant", use_container_width=True, type="primary")

# --- Prediction Logic ---
if run_button:
    if not gene.strip() or not variation.strip():
        st.error("⚠️ Please enter both Gene Name and Variation/Mutation to proceed.", icon="🚨")
    else:
        with st.spinner("🔬 Analyzing genetic variant..."):
            # --- Preprocess and vectorize ---
            gene_proc = gene
            var_proc = variation
            text_proc = nlp_preprocessing(text)

            gene_feat = tfidf_gene.transform([gene_proc])
            var_feat = tfidf_var.transform([var_proc])
            text_feat = tfidf_text.transform([text_proc])
            X = np.hstack([gene_feat.toarray(), var_feat.toarray(), text_feat.toarray()])

            # --- Prediction ---
            pred_class = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            class_labels = [f"Class {i}" for i in range(1, len(proba)+1)]

            # --- Feature presence ---
            gene_vocab = tfidf_gene.get_feature_names_out()
            var_vocab = tfidf_var.get_feature_names_out()
            text_vocab = tfidf_text.get_feature_names_out()

            present_genes = [g for g in gene_vocab if g in gene_proc.lower()]
            present_vars = [v for v in var_vocab if v in var_proc.lower()]
            present_words = [w for w in text_vocab if w in text_proc.split()]

            # --- Interpretability: Top features for predicted class ---
            coefs = model.estimator.coef_[pred_class-1]
            feature_names = np.concatenate([gene_vocab, var_vocab, text_vocab])
            top_indices = np.argsort(coefs)[-10:][::-1]
            top_features = [(feature_names[idx], coefs[idx]) for idx in top_indices]

        # --- Results Display ---
        st.markdown("---")
        st.markdown("## 📊 Classification Results")
        
        st.markdown(f"""
        <div class="result-card">
            <div class="prediction-box">
                <h2 style='color:#667eea; margin-bottom:0.5rem;'>🎯 Predicted Class</h2>
                <h1 style='color:#764ba2; font-size:3.5rem; margin:0.5rem 0;'>Class {pred_class}</h1>
                <p style='font-size:1.4rem; color:#555; margin-top:0.5rem;'>
                    Confidence: <strong style='color:#667eea;'>{100*proba[pred_class-1]:.2f}%</strong>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Detailed Results ---
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 📈 Class Probability Distribution")
            prob_df = pd.DataFrame({
                'Class': class_labels,
                'Probability': proba
            })
            st.bar_chart(prob_df.set_index('Class'))
            
            st.markdown("### 🔍 Input Summary")
            st.markdown(f"""
            <div class="info-card">
                <strong>Gene:</strong> {gene} {'✅' if present_genes else '❌ (Not in training vocabulary)'}<br>
                <strong>Variation:</strong> {variation} {'✅' if present_vars else '❌ (Not in training vocabulary)'}<br>
                <strong>Clinical Text:</strong> {'Provided ✅' if text.strip() else 'Not provided ⚠️'}
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 🧠 Model Interpretation")
            st.markdown("**Top 10 Features Contributing to This Prediction:**")
            
            features_df = pd.DataFrame(top_features, columns=['Feature', 'Coefficient'])
            features_df['Coefficient'] = features_df['Coefficient'].round(4)
            st.dataframe(features_df, use_container_width=True, hide_index=True)
            
            if present_words:
                st.markdown(f"**Clinical text features detected:** {', '.join(present_words[:15])}")
            else:
                st.info("💡 Tip: Adding clinical text can improve prediction accuracy!")

        # --- Additional Information ---
        st.markdown("### ℹ️ Understanding Your Results")
        
        confidence_level = proba[pred_class-1]
        if confidence_level > 0.7:
            confidence_text = "High confidence - The model is quite certain about this classification."
            confidence_color = "#28a745"
        elif confidence_level > 0.4:
            confidence_text = "Moderate confidence - Consider reviewing additional clinical evidence."
            confidence_color = "#ffc107"
        else:
            confidence_text = "Low confidence - This prediction should be interpreted with caution."
            confidence_color = "#dc3545"
        
        st.markdown(f"""
        <div style="background-color:{confidence_color}20; padding:1rem; border-radius:8px; border-left:4px solid {confidence_color};">
            <strong>Confidence Assessment:</strong> {confidence_text}
        </div>
        """, unsafe_allow_html=True)
        
        st.warning("""
        ⚠️ **Important Clinical Disclaimer**: 
        This AI model is a research tool and should NOT be used as the sole basis for clinical diagnosis 
        or treatment decisions. Always consult with qualified healthcare professionals and validate 
        findings through established clinical protocols and additional testing.
        """, icon="⚠️")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div class="footer-section">
    <h3>🏥 Partner Institutions & Resources</h3>
    <p>Collaborating with leading cancer research organizations</p>
</div>
""", unsafe_allow_html=True)

# # --- Partner Logos ---
# col1, col2, col3, col4 = st.columns(4)

# with col1:
#     st.markdown("""
#     <div class="footer-logo-container">
#         <img src="https://i0.wp.com/comacc.org/wp-content/uploads/2024/03/Memorial-Sloan-Kettering-Cancer-Center.png?resize=660%2C140&ssl=1" 
#              style="max-height: 110px; width: auto; object-fit: contain;">
#     </div>
#     <p class="stCaption">Memorial Sloan Kettering</p>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown("""
#     <div class="footer-logo-container">
#         <img src="https://upload.wikimedia.org/wikipedia/commons/7/7e/National_Cancer_Institute_logo.svg"
#              style="max-height: 110px; width: auto; object-fit: contain;">
#     </div>
#     <p class="stCaption">National Cancer Institute</p>
#     """, unsafe_allow_html=True)

# with col3:
#     st.markdown("""
#     <div class="footer-logo-container">
#         <img src="https://cdn.worldvectorlogo.com/logos/kaggle-1.svg" 
#              style="max-height: 110px; width: auto; object-fit: contain;">
#     </div>
#     <p class="stCaption">Kaggle Competition</p>
#     """, unsafe_allow_html=True)

# with col4:
#     st.markdown("""
#     <div class="footer-logo-container">
#         <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRCV78IyDXW3Lsko1cOURswL6GEmlrfB55cPQ&s" 
#              style="max-height: 110px; width: auto; object-fit: contain;">
#     </div>
#     <p class="stCaption">OncoKB Database</p>
#     """, unsafe_allow_html=True)

# st.markdown("<br><br>", unsafe_allow_html=True)
# st.markdown("""
# <div style="text-align:center; padding:1.5rem; background:#f8f9fa; border-radius:10px;">
#     <p style="font-size:0.95rem; color:#666; margin:0;">
#         Made with ❤️ by <a href="https://www.linkedin.com/in/aditya-singh-2b319b299/" target="_blank" style="color:#667eea; text-decoration:none; font-weight:600;">Aditya Singh</a> | 
#         <a href="https://github.com/AdiSinghCodes" target="_blank" style="color:#667eea; text-decoration:none; font-weight:600;">GitHub</a>
#     </p>
# </div>
# """, unsafe_allow_html=True)



