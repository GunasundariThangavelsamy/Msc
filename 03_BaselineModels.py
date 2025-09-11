# BASELINE MODELS
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
ds = load_dataset("KunalEsM/bank_complaint_classifier")
df = ds['train'].to_pandas()

# Prepare data
X = df["Text"]
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

results = {}

# Logistic Regression
name = "Logistic Regression"
model = LogisticRegression(max_iter=500, class_weight="balanced")
print(f"\nTraining {name}...")
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print(f"\n{name} Accuracy: {acc:.4f}\n")
print(classification_report(y_test, y_pred))
results[name] = acc
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.title(f"{name} - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# SVM (LinearSVC)
name = "SVM (LinearSVC)"
model = LinearSVC(class_weight="balanced")
print(f"\nTraining {name}...")
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print(f"\n{name} Accuracy: {acc:.4f}\n")
print(classification_report(y_test, y_pred))
results[name] = acc
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.title(f"{name} - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Random Forest
name = "Random Forest"
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
print(f"\nTraining {name}...")
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print(f"\n{name} Accuracy: {acc:.4f}\n")
print(classification_report(y_test, y_pred))
results[name] = acc
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.title(f"{name} - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
