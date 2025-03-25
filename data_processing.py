import json
import pandas as pd
import numpy as np
import nltk
import re
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK downloads:
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

logging.basicConfig(level=logging.INFO)

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Load JSON data into a DataFrame."""
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            records = []
            for transcript_id, val in data.items():
                record = {
                    'transcript_id': transcript_id,
                    'article_url': val.get('article_url', "Unknown"),
                    'config': val.get('config'),
                    'content': val.get('content'),
                    'conversation_rating': val.get('conversation_rating')
                }
                records.append(record)
            df = pd.DataFrame(records)
            logging.info("Data loaded successfully.")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

class DataCleaner:
    def __init__(self, df):
        self.df = df

    def clean_dataframe(self):
        """Clean the DataFrame by dropping duplicates and handling missing values."""
        # Drop duplicates using only hashable columns to avoid the unhashable 'content' column
        self.df.drop_duplicates(subset=["transcript_id", "article_url", "config"], inplace=True)
        self.df['article_url'] = self.df['article_url'].fillna("Unknown")
        self.df['config'] = self.df['config'].astype('category')
        logging.info("Data cleaning completed.")
        return self.df

class DataTransformer:
    def __init__(self, df):
        self.df = df
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text):
        """Tokenize, remove stopwords, and lemmatize text."""
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [self.lemmatizer.lemmatize(w) for w in tokens if w not in self.stop_words]
        return " ".join(filtered_tokens)

    def transform_content(self):
        """Apply text preprocessing to every message in each transcript."""
        def preprocess_messages(messages):
            for msg in messages:
                original = msg.get('message', '')
                msg['processed_message'] = self.preprocess_text(original)
                msg['message_length'] = len(original.split())
            return messages

        self.df['content'] = self.df['content'].apply(preprocess_messages)
        logging.info("Text preprocessing for content completed.")
        return self.df

    def encode_categories(self):
        """Convert categorical columns into numerical codes."""
        self.df['config_encoded'] = self.df['config'].cat.codes
        logging.info("Categorical encoding completed.")
        return self.df
