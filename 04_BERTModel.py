# BERT MODEL
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset, Dataset as HFDataset, ClassLabel
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
ds = load_dataset("KunalEsM/bank_complaint_classifier")
df = ds['train'].to_pandas()

# Encode target labels
label_encoder = LabelEncoder()
df["label_enc"] = label_encoder.fit_transform(df["label"])
num_labels = len(label_encoder.classes_)
print("Classes:", label_encoder.classes_)
print("Number of labels:", num_labels)

# Tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["Text"], padding="max_length", truncation=True, max_length=128)

dataset = HFDataset.from_pandas(df[["Text", "label_enc"]])
dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label_enc", "labels")
features = dataset.features
features['labels'] = ClassLabel(num_classes=num_labels, names=label_encoder.classes_.tolist())
dataset = dataset.cast(features)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="labels", seed=42)
train_dataset, test_dataset = dataset["train"], dataset["test"]

# Class weights
class_counts = df["label_enc"].value_counts().sort_index().values
class_weights = torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights.sum() / (len(class_weights) * class_weights)
print("Class Weights:", class_weights)

class WeightedBERT(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return {"loss": loss, "logits": logits}

model = WeightedBERT.from_pretrained(model_name, num_labels=num_labels)
model.class_weights = class_weights
model.num_labels = num_labels

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1, "precision": precision, "recall": recall}

import os
os.environ["WANDB_DISABLED"] = "true"
from accelerate import Accelerator
accelerator = Accelerator()

training_args = TrainingArguments(
    output_dir="./bert_results",
    evaluation_strategy="epoch",
    save_total_limit=0,
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=False,
    report_to='none'
)

print ("Training args ready")

import os
os.environ["WANDB_DISABLED"] = "true"
from accelerate import Accelerator
accelerator = Accelerator()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
results = trainer.evaluate()
print("BERT Evaluation:", results)

predictions = trainer.predict(test_dataset)
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)
cm = confusion_matrix(y_true, y_pred, labels=range(num_labels))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("BERT - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
