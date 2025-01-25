# Get Data

# Download kaggle data in Google Colab Upload kaggle json
# The following code block is interactive and designed to be run in a Jupyter environment like Google Colab.
# It includes file upload functionality that will not work directly in a standalone Python script.
# To use this part, run this cell in a Colab notebook.
"""
! pip install -q kaggle
from google.colab import files
files.upload()
"""

! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets download -d 'mohamedbakhet/amazon-books-reviews'

# Create a folder
! mkdir book_data

# Unzip the downloaded .zip compressed file into the book_data folder
! unzip /content/amazon-books-reviews.zip -d book_data

# Data pre
import pandas as pd
import numpy as np
BR = pd.read_csv('/content/book_data/Books_rating.csv')
BR.head()

BR.isnull().sum()

BR = BR.dropna()

BR.isnull().sum()

BR.info()
# Remaining 414548

# Word Segmentation
import nltk
nltk.download('punkt')

#BR['tokenized_text'] = None
from itertools import chain

for index, row in BR.iterrows():

  # If tokenized_text already has a value, skip the current iteration
  if pd.notna(BR.at[index, 'tokenized_text']):
    print("已經有值了") # Already has value
    continue

  df_rating_review_text = row['review/text']
  sentences = nltk.sent_tokenize(df_rating_review_text)
  tokens = [nltk.tokenize.word_tokenize(sent) for sent in sentences]
  # Convert nested list to a single list
  merged_tokens = list(chain.from_iterable(tokens))

  # Put back to dataframe
  BR.at[index, 'tokenized_text'] = merged_tokens

  # Print number sequentially after each loop completes
  print(index + 1)

# Export csv
BR.to_csv('/content/book_data/BR.csv')

# peep
BR.tail()

# Sentiment Analysis
from textblob import TextBlob

text = BR["review/text"][2999953]

# Perform sentiment analysis using TextBlob
blob = TextBlob(text)
sentiment = blob.sentiment

# Sentiment analysis results include two values: polarity and subjectivity
# Polarity: A value between -1 (negative sentiment) and 1 (positive sentiment)
# Subjectivity: A value between 0 (objective) and 1 (subjective)
polarity, subjectivity = sentiment

print(f"Polarity: {polarity}, Subjectivity: {subjectivity}")


#These two values represent two aspects of sentiment analysis results respectively:

#Polarity (polarity): This is a value between -1 and 1, used to measure the sentiment of the text. Polarity value closer to -1 indicates more negative sentiment;
#Polarity value closer to 1 indicates more positive sentiment; polarity value close to 0 indicates neutral sentiment of the text. In this example, the polarity value is 0.155, which means the text has a slight positive sentiment.

#Subjectivity (subjectivity): This is a value between 0 and 1, used to measure the subjectivity of the text. Subjectivity value closer to 0 indicates more objective text;
#Subjectivity value closer to 1 indicates more subjective text. In this example, the subjectivity value is 0.702, which means the text has relatively high subjectivity.

#In summary, in this example, the text has a slight positive sentiment and a relatively high subjectivity.

# Find books with more than 50 reviews
BR_up50 = BR.groupby('Title').filter(lambda x: len(x) > 50)
BR_up50.info()

# Calculate their positive sentiment scores
# Calculate the sentiment score for each review
BR_up50['polarity'] = BR_up50['review/text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Calculate the average sentiment score for each book ID
mean_polarity = BR_up50.groupby('Title')['polarity'].mean()

# Find the top 5 books with the highest positive sentiment score
top_5_positive_books = mean_polarity.nlargest(5)

print(top_5_positive_books)

# Import book data
BD = pd.read_csv('/content/book_data/books_data.csv')
BD.head()

# Convert Pandas Series to DataFrame
top_polarity_df = top_5_positive_books.to_frame().reset_index()

# Rename columns in DataFrame
top_polarity_df.columns = ['Title', 'polarity']

top_polarity_df

# Let's see which are the top 5 books with the best positive sentiment
# Create a list containing all book titles in top_polarity_df
top_titles = top_polarity_df['Title'].tolist()

# Use the isin() function to filter matching rows in original_df
filtered_df = BD[BD['Title'].isin(top_titles)]

filtered_df

# Book covers
from PIL import Image
import urllib.request

for i in filtered_df['image']:

  # Select the image URL to display
  image_url = i

  # Download and display image from URL
  with urllib.request.urlopen(image_url) as url:
      img = Image.open(url)
      img.show()

# tf-idf
import plotly.graph_objects as go
colors = ['gold', 'mediumturquoise','brown']
labels = BR['review/score'].value_counts().keys().map(str)
values = BR['review/score'].value_counts()/BR['review/score'].value_counts().shape[0]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='white', width=0.1)))

fig.show()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
# Use TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', min_df=5, max_df=0.95)
X = vectorizer.fit_transform(BR['review/text'])

# X is the TF-IDF matrix, where each row represents a text and each column represents a word
# This matrix can be used for subsequent text analysis, classification and other tasks

# Assuming you have a target column, such as review/score
y = BR['review/score']

# Use chi-squared test to select the K most influential features
k = 20  # Can be changed as needed
selector = SelectKBest(chi2, k=k)
selector.fit(X, y)

# Get the most influential words
top_words_indices = selector.get_support(indices=True)
top_words = [vectorizer.get_feature_names_out()[index] for index in top_words_indices]

print("Top", k, "important words:")
print(top_words)

# Create a new DataFrame containing only rows where review/score equals 5.0
df_score_5 = BR[BR['review/score'] == 5.0]

# Use TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', min_df=5, max_df=0.95)
X_score_5 = vectorizer.fit_transform(df_score_5['review/text'])

# Calculate the average TF-IDF value for each word
tfidf_means = X_score_5.mean(axis=0)

# Select the top K most important words
k = 10
top_indices = tfidf_means.argsort()[::-1][:k]
top_words = [vectorizer.get_feature_names_out()[index] for index in top_indices]

print("Top", k, "important words for review/score = 5.0:")
print(top_words)

# Create a new DataFrame containing only rows where review/score equals 1.0
df_score_1 = BR[BR['review/score'] == 1.0]

# Use TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', min_df=5, max_df=0.95)
X_score_1 = vectorizer.fit_transform(df_score_1['review/text'])

# Calculate the average TF-IDF value for each word
tfidf_means = X_score_1.mean(axis=0)

# Select the top K most important words
k = 10
top_indices = tfidf_means.argsort()[::-1][:k]
top_words = [vectorizer.get_feature_names_out()[index] for index in top_indices]

print("Top", k, "important words for review/score = 1.0:")
print(top_words)