import pandas as pd
import re
import glob

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    
    # 2. Remove Timestamps (e.g., 29:28)
    text = re.sub(r"\d+:\d+", '', text)
    
    # 3. Remove standalone numbers (e.g., 2030)
    text = re.sub(r"\b\d+\b", '', text)
    
    # 4. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def is_urdu_script(text):
    # Regex range for Arabic/Urdu script letters
    # If the comment contains Urdu script, we will flag it to be deleted
    if re.search(r'[\u0600-\u06FF]', text):
        return True
    return False

def process_files():
    # Find all CSV files in the raw folder
    raw_files = glob.glob('data/raw/*.csv')
    all_cleaned_data =[]

    for file in raw_files:
        df = pd.read_csv(file)
        print(f"Processing {file} - Original rows: {len(df)}")
        
        # Drop empty rows
        df = df.dropna(subset=['text'])
        
        # Apply the text cleaning function
        df['clean_text'] = df['text'].apply(clean_text)
        
        # FILTER 1: Remove rows that contain Urdu script
        df = df[~df['clean_text'].apply(is_urdu_script)]
        
        # FILTER 2: Remove short sentences (less than 3 words like "," or "Nice")
        df = df[df['clean_text'].apply(lambda x: len(str(x).split()) >= 3)]
        
        # Keep only the final cleaned text and add an empty label column
        df = df[['clean_text']].rename(columns={'clean_text': 'text'})
        df['label'] = ""  # Ready for manual labeling!
        all_cleaned_data.append(df)
        
    if all_cleaned_data:
        # Combine all files into one master dataset
        master_df = pd.concat(all_cleaned_data, ignore_index=True)
        
        # Remove duplicate comments
        master_df = master_df.drop_duplicates()
        
        # Save to the processed folder
        output_path = 'data/processed/master_cleaned_data.csv'
        master_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n✅ SUCCESS! Cleaned file saved with {len(master_df)} rows at: {output_path}")
    else:
        print("No raw files found.")

if __name__ == "__main__":
    process_files()