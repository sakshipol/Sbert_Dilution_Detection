# ==========================================
# 1️⃣ Import Libraries
# ==========================================
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sentence_transformers import SentenceTransformer
import seaborn as sns
import matplotlib.pyplot as plt

print("✅ Libraries Imported Successfully")


# ==========================================
# 2️⃣ Load Dataset
# ==========================================
df = pd.read_csv("buzzword_dilution_dataset.csv", encoding="latin1")
print("✅ Dataset Loaded")

X = df["text"].astype(str)
y = df["label"]
print("✅ Features and Labels Prepared")


# ==========================================
# 3️⃣ Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("✅ Data Split Done")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 🔥 MODEL 1: TF-IDF + Naive Bayes
# ==========================================
print("\n==============================")
print("MODEL 1: TF-IDF + Naive Bayes")
print("==============================")

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print("✅ TF-IDF Vectorization Done")

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
print("✅ Naive Bayes Model Trained")

y_pred_nb = nb_model.predict(X_test_tfidf)
print("✅ Prediction Done")

print("\nAccuracy:", accuracy_score(y_test, y_pred_nb))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_nb))

cm_nb = confusion_matrix(y_test, y_pred_nb)
print("✅ Confusion Matrix Generated")

plt.figure()
sns.heatmap(cm_nb, annot=True, fmt="d")
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

cv_nb = cross_val_score(nb_model, X_train_tfidf, y_train, cv=5)
print("✅ Cross Validation Done")
print("5-Fold CV Accuracy:", cv_nb.mean())


# ==========================================
# 🔥 MODEL 2: TF-IDF + SVM
# ==========================================
print("\n==============================")
print("MODEL 2: TF-IDF + SVM")
print("==============================")

svm_model = LinearSVC()
svm_model.fit(X_train_tfidf, y_train)
print("✅ SVM Model Trained")

y_pred_svm = svm_model.predict(X_test_tfidf)
print("✅ Prediction Done")

print("\nAccuracy:", accuracy_score(y_test, y_pred_svm))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_svm))
cm_svm = confusion_matrix(y_test, y_pred_svm)

plt.figure()
sns.heatmap(cm_svm, annot=True, fmt="d")
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


cv_svm = cross_val_score(svm_model, X_train_tfidf, y_train, cv=5)
print("✅ Cross Validation Done")
print("5-Fold CV Accuracy:", cv_svm.mean())


# ==========================================
# 🔥 MODEL 3: SBERT + Logistic Regression
# ==========================================
print("\n==============================")
print("MODEL 3: SBERT + Logistic Regression")
print("==============================")

embedding_file = "embeddings.npy"

# -------- If embeddings already saved --------
if os.path.exists(embedding_file):
    print("📂 Loading saved embeddings...")
    embeddings = np.load(embedding_file)
    print("✅ Embeddings Loaded")

# -------- Else generate once --------
else:
    print("⚡ Generating SBERT embeddings (First Run Only)...")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sbert_model.encode(X.tolist(), batch_size=32, show_progress_bar=True)
    np.save(embedding_file, embeddings)
    print("✅ Embeddings Generated and Saved")

# Split embeddings
X_train_emb = embeddings[X_train.index]
X_test_emb = embeddings[X_test.index]
print("✅ Embeddings Split into Train/Test")

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_emb, y_train)
print("✅ Logistic Regression Trained")

y_pred_lr = lr_model.predict(X_test_emb)
print("✅ Prediction Done")

print("\nAccuracy:", accuracy_score(y_test, y_pred_lr))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_lr))

cm_lr = confusion_matrix(y_test, y_pred_lr)
print("✅ Confusion Matrix Generated")

plt.figure()
sns.heatmap(cm_lr, annot=True, fmt="d")
plt.title("SBERT Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

cv_lr = cross_val_score(lr_model, X_train_emb, y_train, cv=5)
print("✅ Cross Validation Done")
print("5-Fold CV Accuracy:", cv_lr.mean())


# ==========================================
# 📊 Model Comparison Table
# ==========================================
results = pd.DataFrame({
    "Model": ["Naive Bayes", "SVM", "SBERT + LR"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_nb),
        accuracy_score(y_test, y_pred_svm),
        accuracy_score(y_test, y_pred_lr)
    ],
    "CV Accuracy": [
        cv_nb.mean(),
        cv_svm.mean(),
        cv_lr.mean()
    ]
})

print("\n==============================")
print("FINAL MODEL COMPARISON")
print("==============================")
print(results)

print("✅ All Models Executed Successfully")
