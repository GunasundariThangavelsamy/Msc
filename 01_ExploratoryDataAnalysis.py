# DATA OVERVIEW & EXPLORATION
import pandas as pd
# For Excel:
# df = pd.read_excel("YOUR_LINK_OR_PATH_HERE")
# for CSV:
df = pd.read_csv("YOUR_LINK_OR_PATH_HERE")

# Basic info
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords

from datasets import load_dataset
ds = load_dataset("KunalEsM/bank_complaint_classifier")
df = ds['train'].to_pandas()

print("Dataset Shape:", df.shape)
print("\n First 5 rows:")
print("\nFirst 5 rows:")
print(df.head())

print("\n Column Names & Data Types:")
print(df.info())

print("\n Summary Statistics (numerical features):")
print("\nSummary Statistics (numerical features):")
print(df.describe())

# Check for missing values
print("\n Missing Values per Column:")
print(df.isnull().sum())

# Check unique values in each column
print("\n Unique Values per Column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# Target variable distribution
if 'label' in df.columns:
    print("\n Target Variable Distribution:")
    print(df['label'].value_counts())
    sns.countplot(data=df, x='label')
    plt.title("Class Distribution")
    plt.show()

# Exploratory Data Analysis (EDA)
# 1 Class Distribution (Bar Plot)
plt.figure(figsize=(12,6))
sns.countplot(y=df['label'], order=df['label'].value_counts().index, palette="viridis")
plt.title("Class Distribution of Complaint Categories")
plt.xlabel("Count")
plt.ylabel("Category")
plt.show()

# 2 Class Distribution (Interactive Plotly Pie Chart)
fig = px.pie(df, names="label", title="Category Distribution (Interactive Pie)")
fig.show()

# 3 Complaint Length Distribution (Words)
df['word_count'] = df['Text'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(10,5))
sns.histplot(df['word_count'], bins=40, kde=True, color="blue")
plt.title("Distribution of Complaint Text Lengths (in Words)")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.show()

# 4 Complaint Length by Category (Boxplot)
plt.figure(figsize=(14,6))
sns.boxplot(data=df, x='label', y='word_count', palette="Set2")
plt.xticks(rotation=90)
plt.title("Complaint Length Distribution per Category")
plt.show()

# 5 Word Cloud for All Complaints
all_text = " ".join(df['Text'].astype(str))
wordcloud = WordCloud(width=1000, height=600, background_color='white', colormap='viridis').generate(all_text)
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Complaint Texts")
plt.show()

# 6 Top 20 Most Frequent Words
stop_words = set(stopwords.words('english'))
all_words = " ".join(df['Text']).lower().split()
filtered_words = [word for word in all_words if word.isalpha() and word not in stop_words]
word_freq = Counter(filtered_words).most_common(20)
plt.figure(figsize=(12,6))
sns.barplot(x=[x[1] for x in word_freq], y=[x[0] for x in word_freq], palette="magma")
plt.title("Top 20 Most Frequent Words in Complaints")
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.show()

# 7 Heatmap of Category Frequencies (Cross-tab of counts)
category_counts = df['label'].value_counts().reset_index()
category_counts.columns = ['Category', 'Count']
plt.figure(figsize=(10,6))
sns.heatmap(category_counts.set_index('Category').T, annot=True, fmt='d', cmap="YlGnBu")
plt.title("Heatmap of Complaint Categories")
plt.show()
