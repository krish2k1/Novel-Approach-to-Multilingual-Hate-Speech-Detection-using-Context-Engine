# -*- coding: utf-8 -*-
# General-purpose imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split

# NLP-specific imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import list_metrics, load_metric
import torch

# Loading of the t-davidson and /pol/ datasets
tdavidson = pd.read_csv('data/tdavidson_prepared.csv', index_col = [0])
pol = pd.read_csv('data/pol_prepared.csv')

# Loading of metrics for model evaluation
metric_acc = load_metric('accuracy')
metric_f1 = load_metric('f1')
metric_auc = load_metric('roc_auc', "multiclass")

"""# Implementation

### Untrained Model
"""

# Loading of the HateXplain Transformer model and its tokenizer
checkpoint = "Hate-speech-CNERG/bert-base-uncased-hatexplain"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# Confirmation of the model's labels
model.config.id2label

# Splitting of the t-davidson data into training, testing, and validation sets, according to a 60:20:20 split
# The sequences and labels are separated into different variables
train_texts, test_texts, train_labels, test_labels = train_test_split(tdavidson['comment'],
                                                                      tdavidson['label'],
                                                                      test_size=0.2,
                                                                      stratify=tdavidson['label'])

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts,
                                                                    train_labels,
                                                                    test_size=0.25,
                                                                    stratify=train_labels)

# As Transformer models are very heavy on memory, only 250 sequences are evaluated at a time
testcomments = test_texts.tolist()[:250]
testlabels = test_labels.tolist()[:250]

# Implementation of the pre-trained model, without any improvements
tokens = tokenizer(testcomments, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
predictions = torch.nn.functional.softmax(output.logits, dim=-1)
modelresults = pd.DataFrame(predictions.detach().numpy() * 100, columns = ['hate_speech','normal','offensive'])
modelresults

# Creation of a new column that contains, for each row, the column name which had the highest percentage value
modelresults['label'] = modelresults.idxmax(axis=1)

# Transformation of the new column's label contents to their related numerical values
modelresults.loc[modelresults['label'] == 'hate_speech', 'label'] = 0
modelresults.loc[modelresults['label'] == 'normal', 'label'] = 1
modelresults.loc[modelresults['label'] == 'offensive', 'label'] = 2

# Usage of the previously-loaded metrics to evaluate the model
print(metric_acc.compute(predictions = modelresults['label'], references = testlabels))
print(metric_f1.compute(predictions = modelresults['label'], references = testlabels, average="macro"))
print(metric_auc.compute(prediction_scores = predictions, references = testlabels, multi_class='ovr'))

# Deletion of the model-related variables to free up memory
del tokens, output, predictions, modelresults

"""### Fine-tuning Process"""

# Creation of encoding through tokenization for the creation of a Dataset object
train_encodings = tokenizer(train_texts.tolist(), truncation = True, padding = True)
val_encodings = tokenizer(val_texts.tolist(), truncation = True, padding = True)
test_encodings = tokenizer(test_texts.tolist(), truncation = True, padding = True)

# Usage of PyTorch to create a Dataset object to be used for the fine-tuning process of the model
# Encoding and their related labels that were previously separated are again unified
class TFDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TFDataset(train_encodings, train_labels.tolist())
val_dataset = TFDataset(val_encodings, val_labels.tolist())
test_dataset = TFDataset(test_encodings, test_labels.tolist())

### Don't randomly run this code! ###

# Training process of the model, using the training and validation datasets
# The process is very lengthy and may take several hours to complete
training_args = TrainingArguments(
    output_dir = './results',              # output directory
    num_train_epochs = 3,                  # total number of training epochs
    per_device_train_batch_size = 16,      # batch size per device during training
    per_device_eval_batch_size = 64,       # batch size for evaluation
    warmup_steps = 500,                    # number of warmup steps for learning rate scheduler
    weight_decay = 0.01,                   # strength of weight decay
    logging_dir = './logs',                # directory for storing logs
    logging_steps = 10,
)

trainer = Trainer(
    model = model,                         # the instantiated 🤗 Transformers model to be trained
    args = training_args,                  # training arguments, defined above
    train_dataset = train_dataset,         # training dataset
    eval_dataset = val_dataset             # evaluation dataset
)

trainer.train()

### Don't randomly run this code! ###

# Saving of the trained model in the results folder to not waste any work
trainer.save_model('./results')

"""### Trained Model"""

# Loading of the improved HateXplain Transformer model from the results folder
trainedmodel = AutoModelForSequenceClassification.from_pretrained('./results')

# Reconfirmation of the model's labels
model.config.id2label

# Implementation of the improved model
trainedtokens = tokenizer(testcomments, padding=True, truncation=True, return_tensors="pt")
trainedoutput = trainedmodel(**trainedtokens)
trainedpredictions = torch.nn.functional.softmax(trainedoutput.logits, dim=-1)
trainedmodelresults = pd.DataFrame(trainedpredictions.detach().numpy() * 100, columns = ['hate_speech','normal','offensive'])
trainedmodelresults

# Creation of a new column that contains, for each row, the column name which had the highest percentage value
trainedmodelresults['label'] = trainedmodelresults.idxmax(axis=1)

# Transformation of the new column's label contents to their related numerical values
trainedmodelresults.loc[trainedmodelresults['label'] == 'hate_speech', 'label'] = 0
trainedmodelresults.loc[trainedmodelresults['label'] == 'normal', 'label'] = 1
trainedmodelresults.loc[trainedmodelresults['label'] == 'offensive', 'label'] = 2

# Usage of the previously-loaded metrics to evaluate the model
print(metric_acc.compute(predictions = trainedmodelresults['label'], references = testlabels))
print(metric_f1.compute(predictions = trainedmodelresults['label'], references = testlabels, average="macro"))
print(metric_auc.compute(prediction_scores = trainedpredictions, references = testlabels, multi_class='ovr'))

# Deletion of the model-related variables to free up memory
del trainedtokens, trainedoutput, trainedpredictions, trainedmodelresults

"""### Interjection

At this point, both the pre-trained model and its improved version have been tried and tested on the same testing data. The original model with no augmentations performed poorly. The metrics indicated a low accuracy, and from observing the results DataFrame it was visible that the percentage values behind the choices weren't very high either. This means that slight deviations in input could easily tilt the resulting labels in another direction, making the model untrustworthy.

The improved version of the model boasts high scores with the metrics, and it's also very decisive when making a labelling choice. In comparison, the original and the improved versions are like night and day, making the improved version a succesful and trustworthy candidate for the labelling of the /pol/ data.

### Labelling /pol/ Data
"""

### Don't randomly run this code! ###

# As labelling 250 sequences is already very costly in terms of memory, labelling ~450K sequences is unfeasible
# This loop takes batches of 100 sequences every iteration and saves them in a CSV file for later usage
# Memory is freed up between each iteration
# The process is very lengthy and may take a full day to complete
loops = math.ceil(pol.shape[0] / 100)
for i in range(loops):
    start = i * 100
    batch = pol[start: start + 100]

    tokens = tokenizer(batch['comment'].tolist(), padding=True, truncation=True, return_tensors="pt")
    output = trainedmodel(**tokens)
    predictions = torch.nn.functional.softmax(output.logits, dim = -1)
    modelresults = pd.DataFrame(predictions.detach().numpy() * 100, columns = ['hate_speech','normal','offensive'])

    modelresults['label'] = modelresults.idxmax(axis=1)
    modelresults.loc[modelresults['label'] == 'hate_speech', 'label'] = 0
    modelresults.loc[modelresults['label'] == 'normal', 'label'] = 1
    modelresults.loc[modelresults['label'] == 'offensive', 'label'] = 2
    modelresults['post_number'] = pol['post_number'][start: start + 100].tolist()

    modelresults.to_csv('pol_labels.csv', mode = 'a', index = False, header = False)

    del batch, tokens, output, predictions, modelresults

    sys.stdout.write(f"\r loop:{i + 1}/{loops}")
    sys.stdout.flush()

# Loading of the /pol/ labels that were saved in a CSV file
pollabels = pd.read_csv("data/pol_labels.csv")

# Confirming that both datasets have the same shape and that no data is missing
print(pol.shape[0])
print(pollabels.shape[0])

# Merging of the /pol/ dataset and the corresponding labels, based on a common column containing unique post numbers
polmerged = pd.merge(pol, pollabels, on='post_number', how='left')

### Don't randomly run this code! ###

# Saving of the merged data for later usage
polmerged.to_csv('data/pol_final.csv', index=False)