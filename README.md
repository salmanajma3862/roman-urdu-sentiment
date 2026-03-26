# 🇵🇰 Roman Urdu Sentiment Analysis & Code-Mixing (Transformers vs. Classical ML)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)

## 📌 Project Overview
Roman Urdu (Urdu written in the Latin script) is the dominant mode of digital communication in South Asia. However, it presents unique NLP challenges due to heavy **English code-mixing, lack of standard orthography, and Gen-Z internet slang/emojis**. 

This research project introduces a custom-scraped, highly noisy YouTube dataset (7,000+ rows) and compares the performance of Classical Machine Learning (TF-IDF) against state-of-the-art Deep Contextual Embeddings (XLM-RoBERTa). 

### ✨ Key Contributions & Novelty
1. **Custom YouTube Dataset:** Built a raw, real-world dataset of 7,050 comments scraped via the YouTube Data API v3 (focusing on Pakistani dramas and vlogs).
2. **Semi-Supervised Active Learning:** Manually annotated 3,400 rows to establish a verified ground truth, then utilized a baseline model to auto-label the remaining dataset, followed by human verification.
3. **Emoji & Context Preservation:** Intentionally preserved emojis and code-mixed phrases to prove their high semantic value in low-resource language sentiment.
4. **Deep Error Analysis:** Uncovered specific Transformer failure modes in Roman Urdu, specifically regarding "Crying Emoji Paradoxes" and cultural sarcasm.

---

## 📊 Experimental Results

| Model Architecture | Feature Representation | Test Accuracy | F1-Score (Macro) |
| :--- | :--- | :---: | :---: |
| **Logistic Regression** | TF-IDF (Bag of Words) | 70.15% | 70.00% |
| **XLM-RoBERTa (Base)** | Bidirectional Context | **73.80%** | **73.24%** |

*Note: While some literature reports 90%+ accuracies on sterilized Roman Urdu datasets, this project prioritizes real-world robustness. 73.80% represents a highly realistic baseline on messy, heavily code-mixed, and sarcastic internet text.*

---

## 📂 Repository Structure

```text
roman-urdu-sentiment/
│
├── data/
│   ├── raw/                  # Raw YouTube API scrapes
│   └── processed/            # Final fully_labeled_dataset.csv (7050 rows)
│
├── src/
│   ├── preprocessing/
│   │   └── clean.py          # Regex cleaning, emoji preservation, script filtering
│   ├── baselines/
│   │   └── train_baseline.py # TF-IDF + Active Learning auto-labeler
│   └── transformers/
│       └── train_transformer.py # XLM-RoBERTa fine-tuning script
│
├── RESEARCH_DIARY.md         # Detailed logs of methodology, literature review, and error analysis
├── requirements.txt          # Python dependencies
└── README.md

🕵️‍♂️ Error Analysis Highlights
A manual inspection of the XLM-RoBERTa misclassifications revealed that while Transformers beat classical ML, they still struggle with:
Sarcasm disguised by polite markers (e.g., using "Inshallah" or "😊" in an insulting sentence).
Gen-Z Pragmatics, where the loudly crying emoji (😭) denotes extreme laughter/excitement, but the model interprets it strictly as sadness (Negative).
Polite Criticism, where a user mixes heavy praise with a specific demand, causing the model to default to Neutral.
Developed for academic research and publication targeting low-resource NLP.