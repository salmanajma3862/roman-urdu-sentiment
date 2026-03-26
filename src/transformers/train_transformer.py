import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
import evaluate

def compute_metrics(eval_pred):
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
    
    return {"accuracy": accuracy, "f1": f1}

# 1. Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device.upper()} (If this says CPU, stop and change runtime!)")

# 2. Load Data
print("🚀 Loading fully labeled dataset for Transformer...")
# df = pd.read_csv('fully_labeled_dataset.csv') # Colab path
df = pd.read_csv('data/processed/master_cleaned_data.csv') # Local path
print("if you are running in google colab, make sure to upload the master_cleaned_data.csv file to the colab environment and change the path accordingly!")
df['label'] = df['label'].astype(int)
df['text'] = df['text'].astype(str)

# 3. Shuffle & Split Data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
train_size = int(0.8 * len(df))
val_size = int(0.1 * len(df))

train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:train_size+val_size]
test_df = df.iloc[train_size+val_size:]

print(f"📊 Data Split -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# 4. Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# 5. Load Tokenizer & Model
model_name = "xlm-roberta-base"
print(f"🤖 Loading Tokenizer & Model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

print("✂️ Tokenizing datasets...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# 6. Set Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Updated for newest transformers version
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir='./logs',
    logging_steps=50,
    fp16=True, # 🔥 This makes it train 2x faster on GPU!
)

# 7. Train!
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

print("🔥 Starting Training! Grab a coffee...")
trainer.train()

# 8. Evaluate on unseen Test Data
print("\n🏆 Evaluating on completely unseen TEST set...")
test_results = trainer.evaluate(tokenized_test)
print("FINAL TEST METRICS:")
print(test_results)