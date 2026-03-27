# 🧬 Personalized Cancer Diagnosis Using Machine Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://personalised-cancer-diagnosis-9ch8fpcpokqcdha6jkfcss.streamlit.app/)
[![Kaggle Competition](https://img.shields.io/badge/Kaggle-MSK%20Redefining%20Cancer%20Treatment-blue)](https://www.kaggle.com/competitions/msk-redefining-cancer-treatment/overview)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-green)](https://github.com/AdiSinghCodes/Personalised-Cancer-Diagnosis.git)

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Feature Engineering](#-feature-engineering)
- [Model Development](#-model-development)
- [Results & Model Ranking](#-results--model-ranking)
- [Final Solution](#-final-solution)
- [Deployment](#-deployment)
- [Usage](#-usage)
- [Technical Requirements](#-technical-requirements)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## 🎯 Problem Statement

### Competition Organizer
**Memorial Sloan Kettering Cancer Center (MSK)** - A leading cancer treatment and research institution organized the "MSK - Redefining Cancer Treatment" competition as part of NIPS 2017 Competition Track.

### The Challenge
Cancer is caused by genetic mutations in DNA, but not all mutations are harmful:
- **Driver mutations**: Cause cancer and require targeted treatment
- **Passenger mutations**: Do not contribute to cancer development

Currently, molecular pathologists spend enormous amounts of time manually analyzing medical literature to classify genetic variations. This process involves:
1. Selecting genetic variations of interest
2. Searching through medical literature for evidence
3. **Manual classification of mutations (most time-consuming step)**

### Our Mission
Replace the manual classification step with an accurate machine learning model that can:
- Predict mutation classes (1-9) based on gene, variation, and clinical text
- Provide probability estimates for each class
- Offer interpretable results for medical professionals
- **Maintain highest accuracy since we're dealing with human lives**

## 🔬 Project Overview

This project develops a machine learning solution for automated cancer mutation classification using genetic and clinical text data. The system predicts one of 9 classes representing different mutation effects, enabling personalized cancer treatment decisions.

### Key Features
- **Multi-class classification** (9 mutation effect classes)
- **High interpretability** for medical decision-making
- **Probability-based predictions** rather than hard classifications
- **Feature importance analysis** for clinical insights
- **Balanced dataset handling** for minority class representation

## 📊 Dataset

### Data Sources
- **Training Variants**: Gene and variation information
- **Training Text**: Clinical literature descriptions
- **Training Labels**: Ground truth mutation classes (1-9)

### Data Characteristics
- **Imbalanced dataset** with varying class distributions
- **High-dimensional sparse features** after text processing
- **Mixed data types**: Categorical (gene, variation) and text data
- **Missing values**: 5 missing text entries in training data

### Data Preprocessing
- Handled missing text entries
- Verified data uniqueness across variant and text tables
- Created balanced dataset through oversampling for comparison
- Maintained original imbalanced dataset for primary analysis

## 🛠 Methodology

### Performance Metrics
- **Primary Metric**: Multiclass Log-Loss (measures prediction confidence)
- **Secondary Metrics**: Classification accuracy, precision, recall, F1-score
- **Business Requirement**: Low latency not critical; accuracy and interpretability paramount

### Baseline Performance
- Random model baseline: **2.5 log-loss**
- Target: Any sensible model should achieve **< 2.5 log-loss**

## 🔧 Feature Engineering

### 1. Gene Feature Analysis
**Univariate Analysis Results:**
- Genes indexed 0-50: High frequency (common genes)
- Genes > 50: Sharp frequency drop (rare genes)
- **Stability**: 97.14% test data and 96.62% CV data covered by training genes

**Featurization Strategies:**
- **One-Hot Encoding**: Binary representation for each unique gene
  - Best for: Logistic Regression, Linear models
  - Creates sparse, high-dimensional vectors
- **Response Coding**: Probability vectors based on class distributions
  - Best for: Tree-based models (Random Forest)
  - Handles rare genes effectively with Laplace smoothing

**Gene Feature Performance (Logistic Regression):**
- Log-loss: **1.15** (significant improvement over baseline)
- Feature deemed **stable and important**

### 2. Variation Feature Analysis
**Characteristics:**
- **Unstable feature**: Many variations in test/CV not seen in training
- **Limited utility**: Contributes minimally to model performance
- **Usage**: Included in final model but with lower importance weight

### 3. Text Feature Analysis (Most Important)
**Text Processing Pipeline:**

**Bag of Words (BoW) Approach:**
1. Tokenization and preprocessing of clinical text
2. Word frequency counting across entire corpus
3. Document-term matrix creation

**Response Coding for Text:**
1. Calculate word-class co-occurrence probabilities
2. Create probability vectors for each word
3. Apply normalization for fair feature comparison

**Featurization Methods:**
- **One-Hot Encoding**: Binary word presence vectors
- **TF-IDF (Final Choice)**: Term frequency-inverse document frequency
  - Captures word importance relative to document and corpus
  - Handles common vs. rare word weighting
  - **Best performing feature representation**

**Text Feature Performance:**
- **Highest importance** among all features
- Achieves lowest log-loss in univariate analysis
- Final ranking: **Text > Gene > Variation**

## 🤖 Model Development

### Models Implemented & Evaluated

#### 1. Naive Bayes (Baseline Model)
**Variants Tested:**
- **MultinomialNB**: For count-based features (one-hot encoding)
- **GaussianNB**: For continuous features (response coding)
- **ComplementNB**: For imbalanced data handling

**Best Result:** ComplementNB with one-hot encoding
- Train log-loss: 0.8486
- Test log-loss: 1.3071
- CV log-loss: 1.2846
- **Misclassification Rate**: 39%

**Key Insights:**
- ComplementNB handles class imbalance effectively
- Provides clear feature importance through likelihood probabilities
- **High interpretability** for medical decisions

#### 2. K-Nearest Neighbors (KNN)
**Challenges with High-Dimensional Data:**
- Curse of dimensionality with one-hot encoded features
- PCA reduced to 10% variance retention (90% information loss)

**Solution:** Used response-coded features
- **Hyperparameter Tuning**: Grid search over k values
- Applied probability calibration (CalibratedClassifierCV)
- **Performance**: Good log-loss but **poor interpretability**
- **Fatal Flaw**: Complete misclassification of classes 3, 4, 7

#### 3. Logistic Regression (Top Performer)
**Variants Tested:**

**A. Class-Balanced + One-Hot Encoding (Best Overall)**
- Train log-loss: **0.5349**
- Test log-loss: **1.077**
- CV log-loss: **1.086**
- **Misclassification Rate**: 35.52%

**B. Class-Balanced + Response Coding**
- Better log-loss but **completely ignored classes 3 & 8**
- Eliminated due to minority class neglect

**C. Imbalanced + One-Hot Encoding**
- Similar performance to balanced version
- **Failed to predict class 8** (minority class issue)

**Why Logistic Regression Excelled:**
- Linear model suited for high-dimensional sparse features
- L2 regularization prevents overfitting
- `class_weight='balanced'` addresses imbalance effectively
- **Excellent interpretability** through feature coefficients
- Provides reliable probability estimates

#### 4. Linear Support Vector Machine (SVM)
**Configuration:**
- Hinge loss for margin-based classification
- Linear kernel for high-dimensional data and interpretability
- Class-balanced weighting

**Performance:**
- Train log-loss: 0.7186
- Test log-loss: 1.1754
- CV log-loss: 1.1787
- **Misclassification Rate**: 37.03%

**Issues:**
- Poor performance on minority classes (8, 9)
- Confusion between classes 1-4 and 2-7
- **Third position** in model ranking

#### 5. Random Forest
**Configurations Tested:**

**A. Imbalanced Data + One-Hot Encoding (Best RF)**
- **Hyperparameter Tuning**: Grid search (n_estimators: 1000, max_depth: 10)
- Train log-loss: 0.6942
- Test log-loss: 1.1489
- **Misclassification Rate**: 37.40%

**B. Response Coding Version**
- Achieved **lowest log-loss (0.067)** but **highest misclassification (48%)**
- Eliminated due to poor practical performance

**Random Forest Strengths:**
- Excellent feature importance ranking
- Handles mixed data types well
- Good ensemble diversity
- **Second position** in final ranking

#### 6. Ensemble Methods
**A. Stacking Classifier**
- **Base Models**: Logistic Regression, Linear SVM, Multinomial NB
- **Meta-Learner**: Logistic Regression with regularization tuning
- **Performance**: Log-loss 1.044, Misclassification 39.24%
- **Result**: Did not outperform individual models

**B. Voting Classifier**
- Soft voting with probability averaging
- **Performance**: Similar to stacking classifier
- **Conclusion**: Ensemble methods did not provide expected improvement

### Model Performance Summary & Ranking

| Rank | Model | Configuration | Train Loss | Test Loss | CV Loss | Misclass % |
|------|-------|---------------|------------|-----------|---------|------------|
| 🥇 **1** | **Logistic Regression** | Class-balanced + One-hot | **0.5349** | **1.077** | **1.086** | **35.52%** |
| 🥈 **2** | Random Forest | Imbalanced + One-hot | 0.6942 | 1.1489 | 1.1629 | 37.40% |
| 🥉 **3** | Linear SVM | Class-balanced + One-hot | 0.7186 | 1.1754 | 1.1787 | 37.03% |
| 4 | KNN | Response coding | 0.6021 | 1.0698 | 1.074 | 37.21% |
| 5 | Naive Bayes | ComplementNB + One-hot | 0.8486 | 1.307 | 1.284 | 39.09% |

## 🏆 Final Solution

### Advanced Feature Engineering: TF-IDF + Balanced Dataset

**Breakthrough Discovery:**
After comprehensive model testing, we implemented **TF-IDF vectorization with oversampled balanced dataset**, achieving unprecedented performance improvement.

### Final Model: Logistic Regression + TF-IDF + Balanced Data

**Feature Processing:**
- **Text Features**: TF-IDF vectorization capturing term importance
- **Gene Features**: TF-IDF encoding for categorical representation
- **Variation Features**: TF-IDF encoding with appropriate weighting
- **Class Balancing**: Oversampling minority classes to equal representation

**Performance Metrics:**
- **Train Log-loss**: ~0.52
- **Test Log-loss**: ~0.67
- **CV Log-loss**: ~0.64
- **Generalization**: Minimal overfitting gap
- **Stability**: Excellent cross-validation consistency

**Why This Configuration Won:**
1. **TF-IDF superiority**: Better captures feature importance than one-hot encoding
2. **Balanced training**: Ensures all mutation classes receive adequate attention
3. **Optimal regularization**: Prevents overfitting while maintaining performance
4. **Medical relevance**: All 9 classes properly represented in predictions

## 📈 Model Interpretability & Feature Importance

### Critical Business Requirement
Since this model impacts **human lives**, interpretability is non-negotiable:

### Feature Importance Analysis
**Method**: Logistic regression coefficient analysis
- **Positive coefficients**: Increase probability of specific class
- **Negative coefficients**: Decrease probability of specific class
- **Magnitude**: Indicates feature importance level

### Clinical Decision Support
For each prediction, the system provides:
1. **Class probabilities**: Confidence levels for all 9 classes
2. **Top contributing features**: Genes, variations, and text terms driving the prediction
3. **Feature presence validation**: Confirms important features exist in the input
4. **Uncertainty quantification**: Helps doctors assess prediction reliability

### Sample Interpretability Output
```
Predicted Class: 4 (Probability: 0.73)

Top Contributing Features:
- Gene: TP53 (Coefficient: +2.1)
- Text term: "oncogenic" (Coefficient: +1.8)
- Text term: "mutation" (Coefficient: +1.2)
- Variation: R248Q (Coefficient: +0.9)

Alternative Classes:
- Class 1: 0.15
- Class 7: 0.08
- Class 2: 0.04
```

## 🚀 Deployment

### Streamlit Web Application
**Live Demo**: [Personalized Cancer Diagnosis App](https://personalised-cancer-diagnosis-9ch8fpcpokqcdha6jkfcss.streamlit.app/)

### Application Features
1. **Input Interface**: 
   - Gene selection dropdown
   - Variation text input
   - Clinical text description area

2. **Prediction Output**:
   - Primary class prediction with confidence
   - Probability distribution across all 9 classes
   - Feature importance visualization
   - Clinical interpretation guidelines

3. **Model Artifacts**:
   - `tfidf_text_vectorizer.joblib`: Trained text vectorizer
   - `tfidf_gene_vectorizer.joblib`: Trained gene vectorizer  
   - `tfidf_var_vectorizer.joblib`: Trained variation vectorizer
   - `logreg_tfidf_model.joblib`: Final trained model

### Model Testing & Validation
**Testing Strategy:**
- Sample data point validation
- Feature presence verification
- Probability distribution analysis
- Cross-reference with known mutation classifications

## 💻 Usage

### Running Locally

1. **Clone Repository**
```bash
git clone https://github.com/AdiSinghCodes/Personalised-Cancer-Diagnosis.git
cd Personalised-Cancer-Diagnosis
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Launch Application**
```bash
streamlit run frontend.py
```

### Input Format
- **Gene**: Select from dropdown (e.g., "TP53", "BRCA1", "EGFR")
- **Variation**: Enter mutation notation (e.g., "R248Q", "C135Y")
- **Text**: Paste clinical/research text describing the mutation

### Output Interpretation
- **Primary Prediction**: Most likely mutation class
- **Confidence Score**: Probability of primary prediction
- **Alternative Classes**: Other possible classifications
- **Feature Evidence**: Which features support the prediction

## 🔧 Technical Requirements

### Dependencies
```
streamlit>=1.28.0
scikit-learn>=1.3.0
pandas>=1.5.0
numpy>=1.24.0
joblib>=1.3.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.15.0
```

### System Requirements
- **Python**: 3.8+
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 500MB for models and dependencies

## 📁 Project Structure

```
Personalised-Cancer-Diagnosis/
│
├── notebooks/
│   ├── EDA-1.ipynb                 # Initial data exploration
│   ├── EDA-RANDOM-UNIVARITE.ipynb  # Feature analysis
│   ├── Naive-Bayes-3.ipynb         # Baseline model
│   ├── KNN-4.ipynb                 # K-nearest neighbors
│   ├── Logistic-Regression-5.ipynb # Main model development
│   ├── Linear-SVM-6.ipynb          # Support vector machine
│   ├── Random-forest-7.ipynb       # Tree-based model
│   └── Stacking-Voting-8.ipynb     # Ensemble methods
│
├── deployment/
│   ├── frontend.py                 # Streamlit application
│   ├── testing.py                  # Model testing utilities
│   ├── train.py                    # Final model training
│   └── TF-IDF-BALANCED.ipynb       # Advanced feature engineering
│
├── models/
│   ├── tfidf_text_vectorizer.joblib
│   ├── tfidf_gene_vectorizer.joblib
│   ├── tfidf_var_vectorizer.joblib
│   └── logreg_tfidf_model.joblib
│
├── data/
│   ├── training_variants.csv
│   ├── training_text.csv
│   ├── training_variants_balanced.csv
│   └── training_text_balanced.csv
│
└── README.md
```

## 🔮 Future Improvements

### Model Enhancements
1. **Deep Learning Integration**: BERT-based text encoding
2. **Advanced Ensembles**: Weighted voting with confidence-based selection
3. **Active Learning**: Incorporate pathologist feedback for continuous improvement
4. **Uncertainty Quantification**: Bayesian approaches for better confidence intervals

### Feature Engineering
1. **Biological Pathway Integration**: Include gene interaction networks
2. **Literature Mining**: Real-time PubMed integration
3. **Multi-modal Data**: Include imaging and genomic sequence data
4. **Temporal Analysis**: Track mutation evolution over time

### Production Deployment
1. **Clinical Integration**: FHIR-compliant data exchange
2. **Regulatory Compliance**: FDA/CE marking preparation
3. **Scalability**: Kubernetes deployment for hospital systems
4. **Monitoring**: Model drift detection and retraining pipelines

## 🤝 Contributing

We welcome contributions to improve cancer mutation classification:

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/improvement`
3. **Implement changes** with proper documentation
4. **Add tests** for new functionality
5. **Submit pull request** with detailed description

### Contribution Areas
- Model architecture improvements
- Feature engineering techniques
- Clinical validation studies
- User interface enhancements
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Important**: This tool is for research and educational purposes. Clinical decisions should always involve qualified medical professionals.

## 📞 Contact

**Developer**: Aditya Singh
- **GitHub**: [@AdiSinghCodes](https://github.com/AdiSinghCodes)
- **LinkedIn**: [Aditya Singh](https://www.linkedin.com/in/aditya-singh-2b319b299/)
- **Project Repository**: [Personalised Cancer Diagnosis](https://github.com/AdiSinghCodes/Personalised-Cancer-Diagnosis.git)

---

### Acknowledgments

- **Memorial Sloan Kettering Cancer Center** for providing the competition dataset
- **NIPS 2017 Competition Track** for hosting the challenge
- **Kaggle Community** for resources and discussions
- **Open Source Libraries** that made this project possible

---

**⚠️ Medical Disclaimer**: This tool is designed to assist medical professionals and researchers. It should not be used as the sole basis for clinical decisions. Always consult qualified healthcare providers for medical diagnosis and treatment decisions.
