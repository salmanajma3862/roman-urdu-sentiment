# Research Diary: Roman Urdu Sentiment Analysis

## Phase 1: Data Collection & Preprocessing
* **Scraping:** (Why YouTube? Which videos did I target and why?)
youtube comment sections represent the most natural, unfiltered, and highly opinionated Roman Urdu text, providing a rich distribution of positive, negative, and neutral sentiments. I successfully scraped over 8,500 raw comments from popular Pakistani YouTube videos (dramas, news, vlogs, and music).
i choose a drama series called "Meri zindagi hai tu" because it has a very active comment section with a wide variety of sentiments, including praise, criticism, and neutral observations. This diversity is ideal for training a robust sentiment analysis model.
i choose a vlogging channel called "Ducky Bhai" because it has a large and engaged audience that frequently uses Roman Urdu in their comments. The channel's content often elicits strong emotional reactions, making it a rich source of sentiment data. many comments on Ducky Bhai's videos are a mix of praise, criticism, and neutral observations, providing a balanced dataset for sentiment analysis.
* **Preprocessing:** (What did I remove and why? What did I keep and why?)
i removed pure Urdu script (Arabic characters) because my focus is on Roman Urdu, and including pure Urdu would introduce noise and make it harder for the model to learn the specific patterns of Roman Urdu. I also removed URLs, timestamps, and standalone numbers to clean the data and reduce irrelevant information that could distract the model from learning sentiment cues. However, I intentionally kept emojis and code-mixed English/Urdu sentences, as modern transformer models utilize these as strong sentiment signals. Emojis often carry significant emotional weight and can enhance the model's ability to detect sentiment, while code-mixing is a common feature of Roman Urdu communication and can provide valuable context for sentiment analysis.
* **Cleaning Decisions:** 
    * Kept emojis (Why? Because they hold sentiment).
    * Kept code-mixed English (Why? Because it's a strong signal in Roman Urdu).
    * Removed pure English & pure Urdu script (Why? To isolate Roman Urdu).
* **Challenges:** (What was hard about this step?)
manually labbeling data is very time consuming and requires a lot of attention to detail. I had to read each comment carefully and understand the context to label it correctly. Additionally, the presence of sarcasm and complex vocabulary in some comments made it difficult to determine the sentiment accurately.

## Phase 2: Annotation (Labeling)
* **Annotation Rules:** (How did I define Positive, Negative, Neutral? Give examples).
* **The "Active Learning" Shortcut:** 
    * Manually labeled 3,399 rows.
    * Why? To create a verified ground-truth dataset.
    * Used Baseline Model to predict the remaining 3,750 rows. 

## Phase 3: Baseline Models
* **Model Used:** TF-IDF + Logistic Regression (Class weights balanced).
* **Results:** 70.15% Accuracy. 
* **Observations:** Model was good at Positive (0.78 F1) but struggled with Negative (0.65 F1). Why? Probably because of sarcasm and complex vocabulary. 

## Phase 4: Transformer Fine-Tuning
* (To be filled later)

## Phase 5: Error Analysis
* (To be filled later)