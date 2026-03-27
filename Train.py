import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack
import joblib

# 1. Read Data
data = pd.read_csv('training_variants_balanced.csv')
data_text = pd.read_csv('training_text_balanced.csv', sep='\t', names=["ID", "TEXT"], skiprows=1)
result = pd.concat([data, data_text], axis=1)
result.loc[result['TEXT'].isnull(), 'TEXT'] = result['Gene'] + ' ' + result['Variation']

# 2. Split Data
result['Class'] = result['Class'].astype(int)
y = result['Class'].values
X_train, X_temp, y_train, y_temp = train_test_split(result, y, stratify=y, test_size=0.36, random_state=42)
X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5555, random_state=42)
train_df = X_train.copy()
cv_df = X_cv.copy()
test_df = X_test.copy()

# 3. Text Preprocessing
stop_words = set(stopwords.words('english'))
def nlp_preprocessing(text):
    if type(text) is not int:
        text = re.sub('[^a-zA-Z0-9\n]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        return ' '.join([word for word in text.split() if word not in stop_words])
    else:
        return ""
for df in [train_df, cv_df, test_df]:
    df['TEXT'] = df['TEXT'].apply(nlp_preprocessing)

# 4. TF-IDF Feature Extraction
tfidf_text = TfidfVectorizer(min_df=3, max_features=10000)
tfidf_gene = TfidfVectorizer()
tfidf_var = TfidfVectorizer()

train_text_tfidf = tfidf_text.fit_transform(train_df['TEXT'])
cv_text_tfidf = tfidf_text.transform(cv_df['TEXT'])
test_text_tfidf = tfidf_text.transform(test_df['TEXT'])

train_gene_tfidf = tfidf_gene.fit_transform(train_df['Gene'])
cv_gene_tfidf = tfidf_gene.transform(cv_df['Gene'])
test_gene_tfidf = tfidf_gene.transform(test_df['Gene'])

train_var_tfidf = tfidf_var.fit_transform(train_df['Variation'])
cv_var_tfidf = tfidf_var.transform(cv_df['Variation'])
test_var_tfidf = tfidf_var.transform(test_df['Variation'])

train_x = hstack([train_gene_tfidf, train_var_tfidf, train_text_tfidf]).tocsr()
cv_x = hstack([cv_gene_tfidf, cv_var_tfidf, cv_text_tfidf]).tocsr()
test_x = hstack([test_gene_tfidf, test_var_tfidf, test_text_tfidf]).tocsr()

# 5. Balance Training Set with SMOTE
smote = SMOTE(random_state=42)
train_x_bal, y_train_bal = smote.fit_resample(train_x, train_df['Class'].values)

# 6. Train Logistic Regression Model
lr_clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.001, class_weight='balanced', random_state=42)
lr_clf.fit(train_x_bal, y_train_bal)
sig_clf = CalibratedClassifierCV(lr_clf, method="sigmoid")
sig_clf.fit(train_x_bal, y_train_bal)

# 7. Save Model and Vectorizers
joblib.dump(sig_clf, "logreg_tfidf_model.joblib")
joblib.dump(tfidf_text, "tfidf_text_vectorizer.joblib")
joblib.dump(tfidf_gene, "tfidf_gene_vectorizer.joblib")
joblib.dump(tfidf_var, "tfidf_var_vectorizer.joblib")

print("Training complete. Model and vectorizers saved.")