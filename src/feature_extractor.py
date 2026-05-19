"""Feature extraction methods"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np


class FeatureExtractor:
    """Extract features from text data"""
    
    def __init__(self, method='tfidf', max_features=5000):
        """
        Initialize feature extractor
        
        Args:
            method (str): 'tfidf' or 'count'
            max_features (int): Maximum features to extract
        """
        self.method = method
        self.max_features = max_features
        self.vectorizer = None
        self._initialize_vectorizer()
    
    def _initialize_vectorizer(self):
        """Initialize the appropriate vectorizer"""
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2),
                lowercase=True
            )
        elif self.method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2),
                lowercase=True
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def fit(self, texts):
        """
        Fit vectorizer to texts
        
        Args:
            texts (list): List of text documents
            
        Returns:
            self: For method chaining
        """
        self.vectorizer.fit(texts)
        print(f"Vectorizer fitted with {len(self.vectorizer.get_feature_names_out())} features")
        return self
    
    def transform(self, texts):
        """
        Transform texts to feature vectors
        
        Args:
            texts (list): List of texts
            
        Returns:
            sparse matrix: Feature vectors
        """
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """
        Fit and transform in one step
        
        Args:
            texts (list): List of texts
            
        Returns:
            sparse matrix: Feature vectors
        """
        return self.vectorizer.fit_transform(texts)
    
    def get_feature_names(self):
        """
        Get feature names
        
        Returns:
            array: Feature names
        """
        return self.vectorizer.get_feature_names_out()
    
    def get_top_features(self, n=20):
        """
        Get top features by frequency/importance
        
        Args:
            n (int): Number of features
            
        Returns:
            list: Top feature names
        """
        feature_names = self.get_feature_names()
        if len(feature_names) > n:
            return list(feature_names[:n])
        return list(feature_names)
