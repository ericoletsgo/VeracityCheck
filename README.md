# VeracityCheck

# Team Members:

- Eric
- Griffin
- Lydia
- Matthew
- Samantha

# Dataset

![Screenshot 2023-08-17 at 10.35.33 AM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8b84d530-0db6-4f7c-8953-27eb55804e7e/Screenshot_2023-08-17_at_10.35.33_AM.png)

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

**Enhancing Fake News Detection Using Advanced NLP Models**

1. **GPT (Generative Pre-trained Transformer):**
    - Utilize GPT's sequence prediction abilities to analyze and understand the coherence of news articles.
    - Train GPT on a diverse dataset of both genuine and fake news to learn language patterns associated with each.
    - Generate text completions to assess if an article's content aligns with factual information or veers into misleading territory.
    - Leverage GPT's fine-tuned model for classifying articles as genuine or potentially fake.
2. **BERT (Bidirectional Encoder Representations from Transformers):**
    - Fine-tune BERT on a large dataset containing labeled examples of fake and genuine news articles.
    - Leverage BERT's bidirectional context understanding to capture nuanced language patterns that indicate bias or misinformation.
    - Develop a binary classification model using the final BERT embeddings to classify articles as reliable or suspicious.
    - Integrate BERT's understanding of context to identify subtler forms of fake news, such as those using misleading language.
3. **SQuAD (Stanford Question Answering Dataset):**
    - Adapt SQuAD-style question-answering techniques to verify the factual accuracy of news articles.
    - Automatically generate questions based on the content of an article and then extract answers from the article itself.
    - Use the consistency and correctness of extracted answers to gauge the reliability of the article's content.
    - Evaluate the model's ability to accurately answer factual questions about the article's context.
4. **Text Classification Models (e.g., word2vec):**
    - Train text classification models like word2vec on a wide variety of news sources to capture language patterns associated with different perspectives.
    - Use the learned embeddings to assess the similarity of a news article's language with known reliable and unreliable sources.
    - Combine word2vec-based similarity scores with other classification models to enhance the overall fake news detection system.
    - Incorporate sentiment analysis techniques to identify biased language and opinions within news articles.

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

- programming languages:
    - Python
- frameworks & libraries:
    - NLTK
    - Word2Vec
    - Pandas
- web development:
    - gradio
    - CSS
    - HTML
    - Bootstrap
- code editors:
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

# Minimum Viable Product

**Objective:** Develop a basic system that identifies potential fake news articles based on textual features.

1. **Data Collection and Preprocessing:**
    - Gather a small dataset of labeled news articles, including both genuine and fake news examples.
    - Preprocess the text data by tokenizing, removing stopwords, and converting to lowercase.
2. **Feature Extraction:**
    - Extract basic textual features such as word frequency, TF-IDF scores, and n-grams.
3. **Model Building:**
    - Choose a simple classification algorithm (e.g., Naive Bayes or Logistic Regression).
    - Split the dataset into training and testing sets for model evaluation.
4. **Model Training and Evaluation:**
    - Train the chosen classifier on the training dataset using the extracted features.
    - Evaluate the model's performance on the testing dataset using metrics like accuracy, precision, recall, and F1-score.
5. **Baseline Demonstration:**
    - Develop a basic user interface (UI) where users can input a news article's text.
    - Upon submission, the UI should display a binary classification result indicating whether the article is "Likely Genuine" or "Potentially Fake" based on the trained model.
6. **Feedback Collection:**
    - Include a simple feedback mechanism for users to report misclassifications or provide additional context.
7. **Iterative Refinement:**
    - Collect user feedback and misclassification reports to improve the model's accuracy and handling of edge cases.
    - Regularly update the model using new data and refined features.

# Potential Roadblocks

- **Data Quality and Availability:** Might be difficult to obtain a reliable dataset which is reliable of **ALL** types of fake news; resulting in model to struggle to generalize to new, unseen types of misinformation.
- **Fake news strategies may be evolving:** meaning that creators can use different strategies, so the model might be effective against certain tactics but struggle with novel techniques. (**Models might be trained on older datasets, so may not work as well for new**)
- **Model Selection and Tuning:**
    
    Selecting the appropriate model architecture and hyperparameters might require experimentation and tuning to achieve optimal results.
    
    **Explaining Model Decisions:**
    
    - Interpreting why a model classified a certain article as fake or genuine can be challenging, leading to issues of trust and transparency.
    
    **Biased Evaluation Metrics:**
    
    Metrics that don't account for false positives or false negatives might lead to skewed interpretations of model effectiveness.
    
    **Other:**
    
    Issues regarding planning: Not reaching deadline, etc.
    
    Lack of knowledge: more time needed to learn 
    

# Why this Project?

- To promote media literacy skills
- To prevent misinformation through media outlets
- To promote critical thinking among news/media consumers
- To create a better informed generation of voters/citizens

# What do you expect to achieve?

- **Objective:** Develop a system for identifying genuine news articles and distinguishing them from fake news, particularly during the upcoming election season.
- **Importance:** Address the prevalent issue of biased news articles produced by news outlets, which can potentially misinform and influence public opinions.
- **Focus:** Concentrate on building a tool that effectively sifts through news articles to identify inherent biases, helping individuals make more informed decisions.
- **Election Season Emphasis:** Given the propensity for news outlets to produce biased content during election periods, the project's significance is heightened during this time.
- **Outcome:** Aim to create a solution that enhances an individual's ability to differentiate between biased and unbiased news articles, thereby reducing the impact of fake news on their opinions.

By achieving these goals, the project intends to empower people with more accurate information, promoting critical thinking and informed participation in the democratic process.
