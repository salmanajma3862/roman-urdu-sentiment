Research Progress Report: Sentiment Analysis of Roman Urdu Text
Prepared by: Salman Ajmal
Current Phase: Data Collection, Annotation, and Baseline Preparation
1. Introduction & Problem Statement
Roman Urdu (Urdu written in the Latin alphabet) is the primary mode of digital communication in Pakistan and across the South Asian diaspora on social media. However, it remains a heavily "low-resource" language in Natural Language Processing (NLP).
The primary challenges I am addressing include:
Lack of Standard Orthography: Words have multiple spelling variations (e.g., acha, achha, achaa).
Code-Mixing: Users frequently mix pure English, Roman Urdu, and emojis within the same sentence.
Data Scarcity: Existing datasets are either too small, highly imbalanced, or publicly overused (leading to data leakage in modern LLMs).
2. What I Have Done So Far (Methodology)
To ensure high data quality and avoid reusing old Kaggle datasets, I built a custom end-to-end data pipeline:
Step 2.1: Custom Data Scraping
Action: I utilized the YouTube Data API v3 to scrape comments from popular Pakistani/Indian YouTube videos (dramas, news, vlogs, and music).
Why: YouTube comment sections represent the most natural, unfiltered, and highly opinionated Roman Urdu text, providing a rich distribution of positive, negative, and neutral sentiments. I successfully scraped over 8,500 raw comments.
Step 2.2: Data Preprocessing & Cleaning
Action: I wrote a Python script using Regular Expressions (re) to clean the raw data.
Why: I programmatically removed pure Urdu script (Arabic characters), URLs, timestamps, and standalone numbers. However, I intentionally kept emojis and code-mixed English/Urdu sentences, as modern transformer models utilize these as strong sentiment signals.
Step 2.3: PhD-Level Manual Annotation (3,000 Rows Completed)
Action: I designed a strict annotation guideline and have manually labeled 3,000 rows into three classes:
1 (Positive): Praise, prayers (dua), agreement, affection.
0 (Negative): Hate speech, insults, sadness, complaints, and constructive criticism.
2 (Neutral): Factual observations, questions, and demands/requests.
Why: High-quality, human-annotated data is the most critical factor for low-resource NLP. Instead of relying on weak automated labeling, I established a verified ground-truth dataset.
3. Immediate Next Steps (Active Learning & Baselines)
Instead of manually labeling the remaining 5,500 rows, I am implementing an Active Learning / Semi-Supervised approach:
Train a classical baseline model (TF-IDF + Logistic Regression/SVM) on the manually verified 3,000 rows.
Use this model to auto-predict the labels for the remaining 5,500 rows.
Manually review and correct the AI’s predictions to finalize a massive ~8,000+ row dataset efficiently.
Once the dataset is complete, the final phase will involve fine-tuning multilingual transformer models (e.g., XLM-RoBERTa or mBERT) to capture the deep contextual meaning of the Roman Urdu text.
4. Seeking Your Guidance on "Novelty"
While the foundational work (dataset creation and baselines) is solid, I am looking for your expert guidance on how to introduce a strong novel research contribution to make this paper stand out for high-impact journals.
Some initial ideas I have for novelty include:
Code-Mixing Focus: Developing a specialized preprocessing technique or custom tokenization strategy specifically for English-Roman Urdu code-mixing.
Sarcasm Detection: Adding a sub-layer to detect sarcasm in Roman Urdu, which traditional sentiment models fail at.
Cross-Lingual Transfer: Analyzing how well a model trained on pure Hindi or English transfers its weights to this custom Roman Urdu dataset.
I would love to hear your thoughts on these ideas or any other directions you recommend to elevate the scientific value of this work!