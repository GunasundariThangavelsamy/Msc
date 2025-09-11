# TEXT PREPROCESSING & SENTIMENT ANALYSIS
import pandas as pd
from datasets import load_dataset
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords

# Load dataset
print("Loading dataset...")
ds = load_dataset("KunalEsM/bank_complaint_classifier")
df = ds['train'].to_pandas()

# Initialize VADER
print("Initializing VADER sentiment analyzer...")
sia = SentimentIntensityAnalyzer()

# Apply sentiment scoring
df['sentiment_score'] = df['Text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
df['sentiment_label'] = df['sentiment_score'].apply(
    lambda score: 'Positive' if score > 0.05 else ('Negative' if score < -0.05 else 'Neutral')
)

# 8 Distribution of Sentiment Labels
plt.figure(figsize=(6,4))
sns.countplot(x=df['sentiment_label'], palette="coolwarm")
plt.title("Distribution of Sentiment in Complaints")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# 9 Sentiment Distribution Across Categories
plt.figure(figsize=(14,6))
sns.countplot(data=df, x="label", hue="sentiment_label", palette="coolwarm")
plt.xticks(rotation=90)
plt.title("Sentiment Distribution Across Complaint Categories")
plt.xlabel("Complaint Category")
plt.ylabel("Count")
plt.show()

# 10 Sentiment Score Density Plot
plt.figure(figsize=(8,5))
sns.kdeplot(df['sentiment_score'], shade=True, color="purple")
plt.title("Density Distribution of Sentiment Scores")
plt.xlabel("Sentiment Score")
plt.ylabel("Density")
plt.show()
