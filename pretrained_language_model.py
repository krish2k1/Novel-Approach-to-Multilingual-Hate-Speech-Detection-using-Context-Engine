# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

import pandas as pd

# Load the dataset
file_path = 'ar_dataset.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset and its summary information
data_info = data.info()
first_rows = data.head()

data_info, first_rows

import re
from sklearn.model_selection import train_test_split

# Function to clean tweets
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@w+|\#','', text)  # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Apply cleaning function to tweet text
data['clean_tweet'] = data['tweet'].apply(clean_text)

# Splitting dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['clean_tweet'], data['sentiment'], test_size=0.2, random_state=42)

# Show the size of each set
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))

# Fit and transform the training data to create TF-IDF features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data to the same TF-IDF representation
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Show the shape of the resulting TF-IDF matrices
(X_train_tfidf.shape, X_test_tfidf.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Initialize and train the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = log_reg.predict(X_test_tfidf)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

(accuracy, precision, recall, f1)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# Convert to PyTorch dataset
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Load the dataset (Update the path to where your dataset is located)
file_path = 'ar_dataset.csv'
data = pd.read_csv(file_path)

# Assuming 'tweet' is the column with text and 'sentiment' is the label column
data['labels'] = pd.factorize(data['sentiment'])[0]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data['tweet'], data['labels'], test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
model = AutoModelForSequenceClassification.from_pretrained("aubmindlab/bert-base-arabertv2", num_labels=data['labels'].nunique())

train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=128)

train_dataset = TweetDataset(train_encodings, y_train.to_numpy())
test_dataset = TweetDataset(test_encodings, y_test.to_numpy())

!pip install accelerate -U

import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Convert datasets into DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Initialize optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

# Evaluation loop
model.eval()
total_eval_accuracy = 0
for batch in tqdm(test_loader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    total_eval_accuracy += torch.sum(predictions == labels).item()

avg_test_accuracy = total_eval_accuracy / len(test_dataset)
print(f"Test Accuracy: {avg_test_accuracy}")

# Code for real time data url

import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/Israeli%E2%80%93Palestinian_conflict"

response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content using Beautiful Soup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all text elements and concatenate them
    text = ' '.join([element.get_text() for element in soup.find_all(string=True)])

    # Write the extracted text to a txt file
    with open("output.txt", "w", encoding="utf-8") as file:
        file.write(text)

    print("Extraction completed. The output has been saved to output.txt")
else:
    # Print the error message if the request was not successful
    print("Error:", response.status_code, response.text)

# Open the text file in read mode
with open("output.txt", "r", encoding="utf-8") as file:
    # Read the contents of the file
    file_contents = file.read()

# Print the contents of the file
print(file_contents)

pip install langchain