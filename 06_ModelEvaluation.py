# MODEL EVALUATION & COMPARISON
import pandas as pd
import matplotlib.pyplot as plt

# Example results summary (replace with your actual results)
results_summary = pd.DataFrame({
    "Model": ["Logistic Regression", "SVM", "Random Forest", "BERT (base)", "DistilBERT"],
    "Accuracy": [0.9864, 0.9907, 0.9849, 0.9871, 0.9856],
    "Macro F1": [0.986, 0.991, 0.985, 0.987, 0.985]
})

print(results_summary)

# Accuracy Comparison
plt.figure(figsize=(8,5))
plt.bar(results_summary["Model"], results_summary["Accuracy"], color=["skyblue","orange","green","red","purple"])
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=30)
plt.ylim(0.97, 1.0)
plt.show()

# F1 Score Comparison
plt.figure(figsize=(8,5))
plt.bar(results_summary["Model"], results_summary["Macro F1"], color=["skyblue","orange","green","red","purple"])
plt.ylabel("Macro F1 Score")
plt.title("Model F1 Comparison")
plt.xticks(rotation=30)
plt.ylim(0.97, 1.0)
plt.show()
