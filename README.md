# VeracityCheck
## Fake News Detection & Chatbot

## Overview
A Flask-based web application that integrates a BERT-based fake news classifier and a chatbot utilizing Haystack's FAISS document store. The app allows users to check whether a news article is real or fake and interact with an AI-powered chatbot for question answering.

## Features
- **Fake News Detection:** Uses a fine-tuned BERT model to classify news as real or fake.
- **Chatbot with Haystack:** Implements a chatbot that retrieves answers from a FAISS document store using an embedding retriever and a FARM reader.
- **Web Interface:** Flask-based frontend with multiple pages including Home, About, Team, and Chatbot.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip (Python package manager)
- Virtual environment (optional but recommended)
- GPU (optional but recommended for model inference)

### Setup Instructions
1. **Clone the Repository:**
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. **Create a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Required Dependencies
Ensure you have the following Python packages installed:
```bash
pip install flask torch transformers numpy json haystack-ai faiss-cpu
```
If using a GPU, install `faiss-gpu` instead of `faiss-cpu`:
```bash
pip install faiss-gpu
```

## Running the Application
1. **Start the Flask App:**
   ```bash
   python app.py
   ```
2. **Access the Application:**
   Open a browser and go to:
   ```
   http://127.0.0.1:5000
   ```

## File Structure
- `app.py` - Main Flask application
- `templates/` - HTML templates for frontend pages
- `static/` - Static files (CSS, JS, etc.)
- `c3_new_model_weights.pt` - Pretrained BERT model weights for fake news classification
- `news_faiss` - FAISS document store for chatbot retrieval

## Usage
### Fake News Detection
1. Navigate to the homepage (`/`).
2. Enter a news snippet in the input box.
3. Submit the form to receive a classification result (Real or Fake).

### Chatbot
1. Navigate to the chatbot page (`/chatbot`).
2. Enter a query related to news topics.
3. The chatbot retrieves the most relevant answer from the FAISS document store.


# Team Members:

- Eric
- Griffin
- Lydia
- Matthew
- Samantha

# Dataset

<img width="1119" alt="Screenshot 2023-08-17 at 10 35 33 AM" src="https://github.com/user-attachments/assets/80f61dd0-45af-4c93-9ca0-122560dd7645" />
- Two files and eight columns, one file with articles that are fake and another with ones that are true.
- There are four columns in each file:
    1. Title of the article
    2. The content of each article
    3. The main subject of the article (politics news or world news)
    4. The date the article was posted initially (NOT when it was added to the dataset)

**Source:** https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=Fake.csv

# Preprocessing Steps

1. **Feature Selection:**
    - Decide which attributes or characteristics of the text you want to consider as features for your analysis. This could include text content, metadata, and other relevant information.
2. **Tokenize Dataset:**
    - Break down the text into individual words or tokens. This is a fundamental step in text processing.
3. **Remove Stopwords:**
    - Eliminate common words (stopwords) like "the," "and," "is," etc. that might not contribute much to the overall meaning of the text.
4. **Remove Punctuation:**
    - Strip out punctuation marks like commas, periods, and question marks from the text.
5. **Lowercase All Text:**
    - Convert all text to lowercase to ensure consistent comparison of words regardless of their case.
6. **Stem All Text:**
    - Apply stemming to reduce words to their root form. For example, "running" and "runs" would both be reduced to "run."
7. **Lemmatize All Text:**
    - Lemmatization is similar to stemming but aims to reduce words to their base or dictionary form (lemma). For example, "better" becomes "good."
8. **Convert Numbers and Symbols to Words:**
    - Replace numerical values and symbols with their written word equivalents. For instance, "0.1%" could become "zero point one percent."
9. **Remove Accents:**
    - Strip accents from characters, ensuring consistent representation of characters irrespective of diacritics.
10. **Ensuring English Language:**
    - Verify that the text is in English to maintain consistency and avoid processing errors.
11. **Expand Contractions:**
    - Convert contractions like "can't" to "cannot" to ensure uniform representation of words.
12. **Expand Abbreviations:**
    - Convert abbreviations like "etc." to "et cetera" for accurate analysis.
13. **Convert to Word Vectors:**
    - Transform the cleaned text into numerical representations (word vectors) suitable for input to machine learning models. This might involve techniques like TF-IDF or word embeddings like Word2Vec or GloVe.
   

# Models

1. **BERT (Bidirectional Encoder Representations from Transformers):**
    - Fine-tune BERT on a large dataset containing labeled examples of fake and genuine news articles.
    - Leverage BERT's bidirectional context understanding to capture nuanced language patterns that indicate bias or misinformation.
    - Develop a binary classification model using the final BERT embeddings to classify articles as reliable or suspicious.
    - Integrate BERT's understanding of context to identify subtler forms of fake news, such as those using misleading language.
2. **Text Classification Models (e.g., word2vec):**
    - Train text classification models like word2vec on a wide variety of news sources to capture language patterns associated with different perspectives.
    - Use the learned embeddings to assess the similarity of a news article's language with known reliable and unreliable sources.
    - Combine word2vec-based similarity scores with other classification models to enhance the overall fake news detection system.
    - Incorporate sentiment analysis techniques to identify biased language and opinions within news articles.
  
For classification:

• MultiNomial Naive Bayes - classifier algorithm

• **Support Vector Machine - used to train model**

• **Passive Aggressive Classifier - online algorithm that learns from massive streams of data. The idea is to get an example, update the classifier, and throw away the example.**

**Count Vectorizer - The count vectorizer tokenizes a collection of documents and builds a vocabulary of unique words. It can also encode new documents using this vocabulary.**

**Tfidf Transformer - enrich dataset; “term frequency-inverse document frequency”, meaning the weight assigned to each token not only depends on its frequency in a document but also how recurrent that term is in the entire corpora.**

# Evaluation Metrics

- **ROUGE** - Recall-Oriented Understudy for Gisting
- **Cross-validation**
- **F1-score**
    - **Positive Class:** Biased or fake news articles
    - **Negative Class:** Non-biased and genuine news articles
- **Compute Precision and Recall**
    - **Precision**: The ratio of true positives (correctly identified biased/fake news) to the total number of instances predicted as biased/fake news.
    - **Recall**: The ratio of true positives to the total number of actual biased/fake news
    then calculate the f1 score (F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
- **Interpret our F1 score**
A higher F1 score indicates that our model is performing well in terms of both precision and recall.
- **Introduce a ceiling**
Find a bar when it come towards our f1 score, if we repeatedly get above it then our project is good, and if it is below it, we'll obviously make changes

# Tech Stack

- Programming languages:
    - Python
- Frameworks & libraries:
    - NLTK
    - Word2Vec
    - Pandas
- Web development:
    - gradio
    - CSS
    - HTML
    - Bootstrap
- Code editors:
    - google collab.
    - replit

# Basic Outline

- large centralized header/title
- navigation menu
    - about us
    - home
    - detector
    - user feedback?
- user input (URL to news article) box on left side of webpage
    - brief instructions above the box
- output box (fake or real) on right side of webpage
- about us section
    - brief description of each team member
- home page
    - links at bottom
        - AI camp
        - list of resources used (google colab, datasets, etc.)
- brief section on importance of critical thinking & limitations of model below the output on detector page

    

# Why this Project?

- To promote media literacy skills
- To prevent misinformation through media outlets
- To promote critical thinking among news/media consumers
- To create a better informed generation of voters/citizens

## Potential Improvements
- **Enhance Fake News Model:** Improve accuracy by training on a larger dataset.
- **Optimize Chatbot Retrieval:** Implement a more sophisticated ranking mechanism.
- **Deploy Using Docker & Terraform:** Containerize the application for cloud deployment.
