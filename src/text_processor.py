"""Text processing and NLP utilities"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextProcessor:
    """Process and clean text data"""
    
    def __init__(self):
        """Initialize text processor with NLP tools"""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Clean email text
        
        Args:
            text (str): Raw email text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            list: List of tokens
        """
        tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords(self, tokens):
        """
        Remove common stopwords
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Filtered tokens
        """
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        return filtered_tokens
    
    def lemmatize(self, tokens):
        """
        Lemmatize tokens
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Lemmatized tokens
        """
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized
    
    def process(self, text, remove_stops=True, lemmatize=True):
        """
        Complete text processing pipeline
        
        Args:
            text (str): Raw text
            remove_stops (bool): Remove stopwords
            lemmatize (bool): Apply lemmatization
            
        Returns:
            str: Processed text
        """
        # Clean
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        if remove_stops:
            tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        if lemmatize:
            tokens = self.lemmatize(tokens)
        
        # Join back
        processed_text = ' '.join(tokens)
        
        return processed_text
    
    def batch_process(self, texts, **kwargs):
        """
        Process multiple texts
        
        Args:
            texts (list): List of texts
            **kwargs: Arguments for process()
            
        Returns:
            list: Processed texts
        """
        processed = [self.process(text, **kwargs) for text in texts]
        return processed
