import pandas as pd
import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def run_error_analysis():
    print("🔍 Running Error Analysis on Test Data...")

    model_path = "results/best_model"
    data_path = "data/processed/fully_labeled_dataset.csv"

    # 1. Check if the model actually exists locally
    if not os.path.exists(model_path):
        print(f"❌ Error: Could not find saved model at '{model_path}'.")
        print("Please run train_transformer.py first, or download your model from Colab.")
        return

    # 2. Re-load the exact same test dataset
    df = pd.read_csv(data_path)
    df['label'] = df['label'].astype(int)
    df['text'] = df['text'].astype(str)
    
    # Same split as training (Seed 42 ensures we get the exact same test set)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))
    test_df = df.iloc[train_size+val_size:].copy()
    
    test_dataset = Dataset.from_pandas(test_df)

    # 3. Load the Tokenizer and Model
    print("🤖 Loading saved model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Tokenize the test data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # 4. Initialize a simple Trainer just for predicting
    trainer = Trainer(model=model)

    # 5. Get predictions
    print("🧠 Generating predictions...")
    predictions = trainer.predict(tokenized_test)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids

    # 6. Extract the actual text for the test set
    test_texts = test_df['text'].tolist()

    # 7. Create a DataFrame to see where the model failed
    error_df = pd.DataFrame({
        'text': test_texts,
        'true_label': true_labels,
        'predicted_label': predicted_labels
    })

    # Filter only the wrong predictions
    mistakes_df = error_df[error_df['true_label'] != error_df['predicted_label']]
    print(f"❌ The model made {len(mistakes_df)} mistakes out of {len(test_df)} test samples.")

    # Save to CSV
    mistakes_df.to_csv('transformer_mistakes.csv', index=False)
    print("💾 Saved mistakes to 'transformer_mistakes.csv'.")

    # 8. Draw a beautiful Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative (0)', 'Positive (1)', 'Neutral (2)'], 
                yticklabels=['Negative (0)', 'Positive (1)', 'Neutral (2)'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix: XLM-RoBERTa on Roman Urdu')
    
    # Save the image into the results folder
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/confusion_matrix.png', dpi=300)
    print("📊 Confusion Matrix saved as 'results/confusion_matrix.png'.")
    plt.show()

if __name__ == "__main__":
    run_error_analysis()