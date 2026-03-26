# Research Diary: Roman Urdu Sentiment Analysis

## Phase 1: Data Collection & Preprocessing

* **Scraping:**

youtube comment sections represent the most natural, unfiltered, and highly opinionated Roman Urdu text, providing a rich distribution of positive, negative, and neutral sentiments. I successfully scraped over 8,500 raw comments from popular Pakistani YouTube videos (dramas, news, vlogs, and music).
i choose a drama series called "Meri zindagi hai tu" because it has a very active comment section with a wide variety of sentiments, including praise, criticism, and neutral observations. This diversity is ideal for training a robust sentiment analysis model.
i choose a vlogging channel called "Ducky Bhai" because it has a large and engaged audience that frequently uses Roman Urdu in their comments. The channel's content often elicits strong emotional reactions, making it a rich source of sentiment data. many comments on Ducky Bhai's videos are a mix of praise, criticism, and neutral observations, providing a balanced dataset for sentiment analysis.

* **Preprocessing:**

i removed pure Urdu script (Arabic characters) because my focus is on Roman Urdu, and including pure Urdu would introduce noise and make it harder for the model to learn the specific patterns of Roman Urdu. I also removed URLs, timestamps, and standalone numbers to clean the data and reduce irrelevant information that could distract the model from learning sentiment cues. However, I intentionally kept emojis and code-mixed English/Urdu sentences, as modern transformer models utilize these as strong sentiment signals. Emojis often carry significant emotional weight and can enhance the model's ability to detect sentiment, while code-mixing is a common feature of Roman Urdu communication and can provide valuable context for sentiment analysis.

* **Cleaning Decisions:**

I kept emojis because they hold sentiment.
I kept code-mixed English because it's a strong signal in Roman Urdu.
I removed pure English & pure Urdu script because to isolate Roman Urdu.

* **Challenges:**

manually labbeling data is very time consuming and requires a lot of attention to detail. I had to read each comment carefully and understand the context to label it correctly. Additionally, the presence of sarcasm and complex vocabulary in some comments made it difficult to determine the sentiment accurately.

## Phase 2: Annotation (Labeling)

* **Annotation Rules:**

I defined positive, negative and neutral sentiment as follows:
* **Positive:** Comments that express positive sentiment, such as praise, appreciation, or happiness.
* **Negative:** Comments that express negative sentiment, such as criticism, anger, or sadness.
* **Neutral:** Comments that express neutral sentiment, such as observations, questions, or statements of fact.

* **The "Active Learning" Shortcut:** 

I manually labeled 3399 rows and used baseline model to predict the remaining 3750 rows. 

## Phase 3: Baseline Models
* **Model Used:** TF-IDF + Logistic Regression (Class weights balanced).
* **Results:** 70.15% Accuracy. 
* **Observations:** Model was good at Positive (0.78 F1) but struggled with Negative (0.65 F1). Why? Probably because of sarcasm and complex vocabulary. 
model struggled with neutral as well. when i was manually seeing the data labeled by baseline model and checking if baseline model did any mistake, i found out an example like that 

"Yeh drama bohot aj kl chl raha hai sirf name ki waja sai open Kiya hai AJ ramzan hai inshallah abi papers ki tyri krraha huwn us Kai bd dekhuwn ga jab 2nd yr Kai papers de kr"

the baseline model had labelled it as negative but it is actually neutral. the comment is basically saying that the person is waiting for the second episode of the drama to come out so that they can watch it after their exams. there is no positive or negative sentiment in this comment, it is just a neutral statement about waiting for something. this is an example of how sarcasm and complex vocabulary can make it difficult for the model to accurately predict sentiment.

another example where baseline model failed was

yeh car konsi hn plz batado ❤

its clearly a neutral comment because the person is asking about the car in the video and there is no positive or negative sentiment in this comment. but the baseline model had labelled it as negative. i would not be surprised if model had labelled it a positive comment because of the heart emoji at the end. this is an example of baseline model getting it completely wrong.

another one

Ye cahe Spotify se htt gya pr mere dil se nhi htt paya 😭😭😭🎀🎀🤭🤭💗💓💓💞💞🫀💕💕❤️❤️💗,0.0

model had labelled it as negative maybe because of the crying emojis but it is actually a positive comment because the person is expressing their love for the song and how they will miss it on Spotify. this is another example of how emojis can be a strong signal for sentiment and how the model can get it wrong if it does not understand the context.



## Phase 4: Transformer Fine-Tuning

* **Model Used:** XLM-RoBERTa (base) via Hugging Face.
* **Hardware:** NVIDIA T4 GPU (Google Colab).
* **Hyperparameters:** 3 Epochs, Learning Rate 2e-5, Batch Size 16.
* **Results:** 
    * Test Accuracy: 73.80%
    * Test F1-Score: 73.24%
* **Scientific Observation:** The deep learning model outperformed the classical TF-IDF baseline by ~3.65%. This proves that contextual embeddings (reading sentences bi-directionally) are superior for handling the complex syntax, code-mixing, and sarcasm present in Roman Urdu.

## Phase 5: Error Analysis



## Phase 6: Literature Review

While some recent studies (e.g., Nikhar Azhar, 2022) report anomalous perfect accuracies (100%) using DistilBERT on Roman Urdu, such metrics often indicate data leakage or evaluation on highly sterilized, non-representative datasets. In contrast, this study prioritizes real-world robustness. By evaluating XLM-RoBERTa on a newly scraped, highly noisy, and subjectively complex YouTube dataset, we achieve a scientifically realistic accuracy of 73.80%, proving that modern LLMs still struggle with Gen-Z code-mixing and cultural sarcasm."