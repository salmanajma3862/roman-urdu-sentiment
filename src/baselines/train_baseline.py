import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def run_active_learning():
    print("🚀 Loading dataset...")
    # Load the dataset
    df = pd.read_csv('data/processed/master_cleaned_data.csv')
    
    # Clean up the label column (convert to numeric, push empties to NaN)
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    
    # Split into Labeled (the 3000 you did) and Unlabeled (the rest)
    labeled_df = df[df['label'].notna()].copy()
    unlabeled_df = df[df['label'].isna()].copy()
    
    print(f"✅ Manually Labeled rows: {len(labeled_df)}")
    print(f"⏳ Unlabeled rows remaining: {len(unlabeled_df)}\n")
    
    # Check class distribution
    print("📊 Class Distribution in your Labeled Data:")
    print(labeled_df['label'].value_counts())
    print("-" * 30)

    # --- PART 1: TRAIN BASELINE MODEL ---
    print("\n🧠 Training Baseline Model (TF-IDF + Logistic Regression)...")
    
    X = labeled_df['text'].astype(str)
    y = labeled_df['label'].astype(int)
    
    # Split into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert text to numbers using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_vec, y_train)
    
    # Evaluate the model (This goes in your paper!)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n🏆 BASELINE ACCURACY: {acc * 100:.2f}%\n")
    print("Detailed Metrics (Save this for your paper):")
    print(classification_report(y_test, y_pred))
    
    # --- PART 2: AUTO-LABEL THE REST ---
    if len(unlabeled_df) > 0:
        print(f"\n🪄 Auto-labeling the remaining {len(unlabeled_df)} rows...")
        X_unlabeled = unlabeled_df['text'].astype(str)
        X_unlabeled_vec = vectorizer.transform(X_unlabeled)
        
        # Predict labels
        auto_predictions = model.predict(X_unlabeled_vec)
        unlabeled_df['label'] = auto_predictions
        
        # Combine everything back together
        final_df = pd.concat([labeled_df, unlabeled_df], ignore_index=True)
        
        # Save the new fully labeled dataset
        output_path = 'data/processed/fully_labeled_dataset.csv'
        final_df.to_csv(output_path, index=False)
        print(f"\n✅ SUCCESS! Fully labeled dataset saved to: {output_path}")
        print("Next step: Just quickly scroll through the new file to fix any AI mistakes!")

if __name__ == "__main__":
    run_active_learning()