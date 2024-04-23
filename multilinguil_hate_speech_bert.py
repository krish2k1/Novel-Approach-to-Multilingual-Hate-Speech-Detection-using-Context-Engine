# -*- coding: utf-8 -*-
!pip install seaborn emoji datasets

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import emoji
import tensorflow as tf
import tf_keras
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datasets import Dataset, DatasetDict, concatenate_datasets

# resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
# On TPU VMs use this line instead:
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)

# strategy = tf.distribute.TPUStrategy(resolver)
# For testing without a TPU use this line instead:
# strategy = tf.distribute.OneDeviceStrategy("/cpu:0")

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

np.object = object

model_name = "google-bert/bert-base-multilingual-cased"

pandas_dataset = pd.read_csv("/kaggle/input/multilingual-hatespeech-dataset/Dataset/Training/MultiLanguageTrainDataset.csv", usecols=["text", "label", "language"])
pandas_dataset.head()

removed = pandas_dataset.drop(pandas_dataset[pandas_dataset.language==8].index)

sns.countplot(removed, x="label")
plt.show()

sns.countplot(removed, x="language")
plt.show()

removed["text_cleaned"] = removed["text"].str.replace(r"@[\d\w_]+\s?", "", regex=True)
removed["text_cleaned"] = removed["text_cleaned"].str.replace(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", "", regex=True)
removed["text_cleaned"] = removed["text_cleaned"].str.replace(r"pic.twitter.com/[\w\d]+", "", regex=True)
removed["text_cleaned"] = removed["text_cleaned"].apply(lambda x:emoji.demojize(x))

removed.head()

ds = None
for language in removed["language"].unique():
    train_val_test = Dataset.from_pandas(removed[removed.language==language]).train_test_split(train_size=0.7)
    val_test = train_val_test["test"].train_test_split(train_size=2/3)
    if ds is None:
        ds = DatasetDict({
            "train": train_val_test["train"],
            "valid": val_test["train"],
            "test": val_test["test"]
        })
    else:
        ds["train"] = concatenate_datasets([ds["train"], train_val_test["train"]])
        ds["valid"] = concatenate_datasets([ds["valid"], val_test["train"]])
        ds["test"] = concatenate_datasets([ds["test"], val_test["test"]])

ds

ds_rem = ds.remove_columns(["text", "language", "__index_level_0__"])
ds_rem

tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Vocab size is {tokenizer.vocab_size}")
print(f"Max length is {tokenizer.model_max_length}")
print(f"Model input names are {tokenizer.model_input_names}")

tok_dataset = ds_rem.map(lambda ds_t:tokenizer(ds_t["text_cleaned"], padding="max_length", truncation=True), batched=True)

tok_dataset = tok_dataset.remove_columns(["text_cleaned"])

train_set = tok_dataset["train"].with_format("tf")
valid_set = tok_dataset["valid"].with_format("tf")
test_set = tok_dataset["test"].with_format("tf")

train_features = {x:train_set[x] for x in tokenizer.model_input_names}
train_set_final = tf.data.Dataset.from_tensor_slices((train_features, train_set["label"])).shuffle(len(train_set)).batch(8)
test_features = {x:test_set[x] for x in tokenizer.model_input_names}
test_set_final = tf.data.Dataset.from_tensor_slices((test_features, test_set["label"])).shuffle(len(test_set)).batch(8)
valid_features = {x:valid_set[x] for x in tokenizer.model_input_names}
valid_set_final = tf.data.Dataset.from_tensor_slices((valid_features, valid_set["label"])).shuffle(len(valid_set)).batch(8)

# gpus = tf.config.list_physical_devices('GPU')
# if len(gpus) < 2:
#     raise RuntimeError("This example requires at least two GPUs.")

# strategy = tf.distribute.MirroredStrategy()

# with strategy.scope():
#     model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
#     model.compile(
#         optimizer = "adam",
#         metrics = ["acc"]
#     )

with strategy.scope():
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.compile(
        optimizer = "adam",
        loss = "binary_crossentropy",
        metrics = ["acc"]
    )

# model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
# model.compile(
#     optimizer = "adam",
#     loss = "binary_crossentropy",
#     metrics = ["acc"]
# )

model_checkpoint_callback = tf_keras.callbacks.ModelCheckpoint(
    filepath="/kaggle/working/checkpoint_model.keras",
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

model.summary()

history = model.fit(train_set_final, validation_data=valid_set_final, epochs=5)









