# define a function to get the TF-IDF distribution for words
import pandas as pd 
from collections import Counter 
from nltk.corpus import stopwords  
import nltk  
import re
# download the stopwords 
nltk.download('stopwords')

def calculate_word_frequencies(df, text_column='text'):
    stop_words = set(stopwords.words('english'))

    # Function to preprocess and tokenize a document
    def tokenize(text):
        # Convert text to lowercase and split into words (tokens)
        tokens = text.lower().split()
        # Remove stopwords
        tokens = [word for word in tokens if word not in stop_words]
        return tokens

    # Apply the tokenization to each document
    df['tokens'] = df[text_column].apply(tokenize)

    # Count word frequencies per document
    df['word_freq'] = df['tokens'].apply(Counter)

    return df

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def clean_and_normalize_text(df, column_name='content'):
    # Set of stopwords
    stop_words = set(stopwords.words('english'))
    
    # Initialize stemmer and lemmatizer
    # stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Function to clean and normalize a single piece of text
    def process_document(doc):
        # Remove punctuation and digits
        doc = re.sub(r'[\d]|[^\w\s]', ' ', doc)

        # Tokenize text
        tokens = word_tokenize(doc.lower())

        # Remove stopwords, then stem and lemmatize tokens
        processed_tokens = [
            lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
        ]

        # Join tokens back into a string
        return ' '.join(processed_tokens)

    # Apply the processing function to the specified column
    df[column_name] = df[column_name].apply(process_document)

    return df

import pandas as pd
from collections import Counter

def aggregate_daily_word_frequencies(df):
    # Group the DataFrame by 'date'
    grouped = df.groupby('date')
    # Aggregate the word frequency dictionaries by date
    def aggregate_dicts(series):
        total_count = Counter()
        for dictionary in series:
            total_count.update(dictionary)
        return dict(total_count)
    # Apply the aggregation function to the 'word_freq' column
    daily_word_freq = grouped['word_freq'].agg(aggregate_dicts).reset_index()

    return daily_word_freq

def convert_to_long_format(df):
    """
    Converts a DataFrame from wide format (date, word_freq dictionary) to 
    long format (date, word_id, word_freq).
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns 'date' and 'word_freq', where
                       'word_freq' is a dictionary with words as keys and
                       their counts as values.
                       
    Returns:
    pd.DataFrame: A long-format DataFrame with columns 'date', 'word_id', and 'word_freq'.
    """
    # Create a new DataFrame that includes the date expanded with word_id and word_freq columns
    long_df = df.set_index('date')['word_freq'].apply(pd.Series).stack().reset_index()
    long_df.columns = ['date', 'word_id', 'word_freq']  # Rename columns appropriately
    
    return long_df