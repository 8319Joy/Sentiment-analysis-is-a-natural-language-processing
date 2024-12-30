#!/usr/bin/env python
# coding: utf-8
Sentiment analysis is a natural language processing (NLP) technique used to determine the sentiment expressed in a piece of text—whether the sentiment is positive, negative, or neutral. It is widely used in various applications, such as social media monitoring, customer feedback analysis, and market research, to gauge public opinion or customer sentiment towards products or brands.

In Python, sentiment analysis can be performed using several libraries, including TextBlob, VADER, spaCy, and machine learning frameworks like scikit-learn or transformers from Hugging Face. Below are examples of how to conduct sentiment analysis using different tools.
# # Example 1: Using TextBlob
# TextBlob is a simple library for processing textual data that provides a simple API for diving into common natural language processing (NLP) tasks.
# In[35]:


from textblob import TextBlob
import numpy

# Sample texts
texts = [
    "I love this product! It's amazing.",
    "This is the worst experience I've ever had.",
    "It's okay, neither good nor bad."
]

# Analyzing sentiment
for text in texts:
    blob = TextBlob(text)
    sentiment = blob.sentiment
    print(f"Text: '{text}'")
    print(f"Sentiment: {sentiment}\n")  # Contains polarity and subjectivity


# # Example 2: Using VADER from NLTK
# VADER (Valence Aware Dictionary and sEntiment Reasoner) is a specialized library within NLTK that is effective for sentiment analysis, particularly in social media contexts.

# In[36]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Sample texts
texts = [
    "I'm so happy and excited!",
    "This is a terrible nightmare.",
    "What just happened was unexpected."
]

# Analyzing sentiment
for text in texts:
    score = analyzer.polarity_scores(text)
    print(f"Text: '{text}'")
    print(f"Scores: {score}\n")  # pos, neg, neu and compound scores


# # Example 3: Using Pre-trained Models (Transformers)
# For more complex analyses, you can use pre-trained models from the transformers library by Hugging Face. Here’s an example using a sentiment analysis model.

# In[47]:


pip install tf-keras


# In[9]:


pip install transformers


# In[38]:


pip install tensorflow==2.12


# # Example 3: Using Pre-trained Models (Transformers)
# For more complex analyses, you can use pre-trained models from the transformers library by Hugging Face. Here’s an example using a sentiment analysis model.

# In[44]:


pip install transformers


# In[49]:


from transformers import pipeline


# # Load a Pre-trained Sentiment Analysis Model
# The pipeline function simplifies loading pre-trained models. Initialize it for sentiment analysis:

# In[48]:


# Import the pipeline function
from transformers import pipeline

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis')

# Use the pipeline on some text
results = sentiment_pipeline("I love using Hugging Face's transformers library!")
print(results)
# Analyze sentiment of text
texts = [
    "I love using Hugging Face Transformers, it's amazing!",
    "I had a terrible experience with this product.",
    "The movie was just okay, nothing special."
]

# Get predictions
results = sentiment_pipeline(texts)

# Print the results
for i, result in enumerate(results):
    print(f"Text: {texts[i]}")
    print(f"Sentiment: {result['label']}, Confidence: {result['score']:.2f}\n")


# # Example 4: Custom Model using Scikit-learn
# You can also build a custom sentiment analysis model using machine learning libraries like Scikit-learn. Here’s a simple example using logistic regression.

# In[14]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Sample dataset
data = {
    'text': ["I love this!", "This is awful.", "Best thing ever!", "I hate it.", "It's okay."],
    'sentiment': [1, 0, 1, 0, 0]
}
df = pd.DataFrame(data)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Creating a model pipeline
model = make_pipeline(CountVectorizer(), LogisticRegression())

# Training the model
model.fit(X_train, y_train)

# Predicting on test data
predictions = model.predict(X_test)

for text, pred in zip(X_test, predictions):
    print(f"Text: '{text}' | Predicted Sentiment: {pred}\n")


# # Summary
# Sentiment analysis can be implemented in various ways depending on the complexity required. Libraries like TextBlob and VADER are great for quick analyses, while Hugging Face Transformers provide state-of-the-art accuracy with deep learning models. Alternatively, building a custom model with Scikit-learn allows for more tailored approaches, according to specific datasets and requirements.

# In[ ]:




