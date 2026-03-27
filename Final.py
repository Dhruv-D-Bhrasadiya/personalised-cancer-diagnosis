import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, confusion_matrix
from sklearn.preprocessing import normalize
from tabulate import tabulate
import seaborn as sns
import re
import joblib

# 1. Data Loading
data = pd.read_csv('training_variants')
data_text = pd.read_csv('training_text', sep=r'\|\|', engine='python', names=["ID", "TEXT"], skiprows=1)
result = pd.merge(data, data_text, on='ID', how='left')
result.loc[result['TEXT'].isnull(), 'TEXT'] = result['Gene'] + ' ' + result['Variation']

# 2. Preprocessing: Lowercase gene/variation/text for consistency
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

result['Gene'] = result['Gene'].str.lower()
result['Variation'] = result['Variation'].str.lower()

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

result['TEXT'] = result['TEXT'].apply(nlp_preprocessing)

# 3. Train/Validation/Test Split
y_true = result['Class'].values
X_train, test_df, y_train, y_test = train_test_split(result, y_true, stratify=y_true, test_size=0.2, random_state=42)
train_df, cv_df, y_train, y_cv = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=42)

# 4. One-hot Encoding
gene_vectorizer = CountVectorizer(lowercase=True)
variation_vectorizer = CountVectorizer(lowercase=True)
text_vectorizer = CountVectorizer(min_df=3, lowercase=True)

train_gene_feature_onehotCoding = gene_vectorizer.fit_transform(train_df['Gene'])
test_gene_feature_onehotCoding = gene_vectorizer.transform(test_df['Gene'])
cv_gene_feature_onehotCoding = gene_vectorizer.transform(cv_df['Gene'])

train_variation_feature_onehotCoding = variation_vectorizer.fit_transform(train_df['Variation'])
test_variation_feature_onehotCoding = variation_vectorizer.transform(test_df['Variation'])
cv_variation_feature_onehotCoding = variation_vectorizer.transform(cv_df['Variation'])

train_text_feature_onehotCoding = text_vectorizer.fit_transform(train_df['TEXT'])
test_text_feature_onehotCoding = text_vectorizer.transform(test_df['TEXT'])
cv_text_feature_onehotCoding = text_vectorizer.transform(cv_df['TEXT'])

# Normalize text features
train_text_feature_onehotCoding = normalize(train_text_feature_onehotCoding, axis=0)
test_text_feature_onehotCoding = normalize(test_text_feature_onehotCoding, axis=0)
cv_text_feature_onehotCoding = normalize(cv_text_feature_onehotCoding, axis=0)

# Combine all features
from scipy.sparse import hstack
train_gene_var_onehotCoding = hstack((train_gene_feature_onehotCoding, train_variation_feature_onehotCoding))
test_gene_var_onehotCoding = hstack((test_gene_feature_onehotCoding, test_variation_feature_onehotCoding))
cv_gene_var_onehotCoding = hstack((cv_gene_feature_onehotCoding, cv_variation_feature_onehotCoding))

train_x_onehotCoding = hstack((train_gene_var_onehotCoding, train_text_feature_onehotCoding)).tocsr()
test_x_onehotCoding = hstack((test_gene_var_onehotCoding, test_text_feature_onehotCoding)).tocsr()
cv_x_onehotCoding = hstack((cv_gene_var_onehotCoding, cv_text_feature_onehotCoding)).tocsr()

train_y = np.array(list(train_df['Class']))
test_y = np.array(list(test_df['Class']))
cv_y = np.array(list(cv_df['Class']))

# 5. Hyperparameter Tuning and Model Training
alpha = [10 ** x for x in range(-6, 3)]
cv_log_error_array = []
for i in alpha:
    print("for alpha =", i)
    clf = SGDClassifier(class_weight='balanced', alpha=i, penalty='l2', loss='log_loss', random_state=42)
    clf.fit(train_x_onehotCoding, train_y)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_x_onehotCoding, train_y)
    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)
    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_))
    print("Log Loss :", log_loss(cv_y, sig_clf_probs))

fig, ax = plt.subplots()
ax.plot(alpha, cv_log_error_array, c='g')
for i, txt in enumerate(np.round(cv_log_error_array, 3)):
    ax.annotate((alpha[i], str(txt)), (alpha[i], cv_log_error_array[i]))
plt.grid()
plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha i's")
plt.ylabel("Error measure")
plt.show()

best_alpha = np.argmin(cv_log_error_array)
clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='log_loss', random_state=42)
clf.fit(train_x_onehotCoding, train_y)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_x_onehotCoding, train_y)

predict_y = sig_clf.predict_proba(train_x_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:", log_loss(train_y, predict_y, labels=clf.classes_))
predict_y = sig_clf.predict_proba(cv_x_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:", log_loss(cv_y, predict_y, labels=clf.classes_))
predict_y = sig_clf.predict_proba(test_x_onehotCoding)
print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:", log_loss(test_y, predict_y, labels=clf.classes_))

# 6. Feature Importance for Multiple Test Points

# ... (all your previous imports and code up to get_imp_feature_names) ...

def get_imp_feature_names(indices, text, gene, variation, no_feature, removed_ind = [], predicted_cls=None):
    word_present = 0
    tabulte_list = []
    incresingorder_ind = 0
    gene = gene.lower()
    variation = variation.lower()
    text_words = set(text.split())
    gene_features = gene_vectorizer.get_feature_names_out()
    variation_features = variation_vectorizer.get_feature_names_out()
    text_features = text_vectorizer.get_feature_names_out()

    # Print preprocessed text words for debugging
    print("Preprocessed text words:", text_words)

    # Collect top text features for the predicted class
    top_text_features = []
    for i in indices:
        if i >= len(gene_features) + len(variation_features):
            idx = i - (len(gene_features) + len(variation_features))
            if idx < len(text_features):
                top_text_features.append(text_features[idx])
    print("Top text features for this class:", top_text_features)

    for i in indices:
        if i < len(gene_features):
            present = (gene_features[i] == gene)
            tabulte_list.append([incresingorder_ind, "Gene", gene_features[i], present])
        elif i < len(gene_features) + len(variation_features):
            idx = i - len(gene_features)
            present = (variation_features[idx] == variation)
            tabulte_list.append([incresingorder_ind, "Variation", variation_features[idx], present])
        else:
            idx = i - (len(gene_features) + len(variation_features))
            if idx < len(text_features):
                word = text_features[idx]
                yes_no = word in text_words
                if yes_no:
                    word_present += 1
                tabulte_list.append([incresingorder_ind, "Text", word, yes_no])
            else:
                tabulte_list.append([incresingorder_ind, "Text", "Index out of bounds", False])
        incresingorder_ind += 1
    print(word_present, "most important text features are present in our query point")
    print("-"*50)
    if predicted_cls is not None:
        print("The features that are most important for the ", predicted_cls[0], " class:")
    print(tabulate(tabulte_list, headers=["Index", 'Feature type', 'Feature name', 'Present in input?']))

num_samples = 5  # Number of test samples to check
np.random.seed(42)
sample_indices = np.random.choice(len(test_df), num_samples, replace=False)

for idx in sample_indices:
    predicted_cls = sig_clf.predict(test_x_onehotCoding[idx])
    print("="*60)
    print(f"Test Sample Index: {idx}")
    print("Actual Class :", test_y[idx])
    print("Predicted Class :", predicted_cls[0])
    print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[idx]),4))
    indices = np.argsort(-1*abs(clf.coef_))[predicted_cls-1][:,:20]
    print("-"*50)
    get_imp_feature_names(
        indices[0],
        test_df['TEXT'].iloc[idx],
        test_df['Gene'].iloc[idx],
        test_df['Variation'].iloc[idx],
        20
    )


# Save model and vectorizers for deployment
joblib.dump(sig_clf, "final_lr_model.pkl")
joblib.dump(gene_vectorizer, "gene_vectorizer.pkl")
joblib.dump(variation_vectorizer, "variation_vectorizer.pkl")
joblib.dump(text_vectorizer, "text_vectorizer.pkl")

print("Model and vectorizers saved. Ready for frontend integration.")