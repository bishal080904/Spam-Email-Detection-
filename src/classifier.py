"""Machine learning classifiers for spam detection"""

import pickle
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from .text_processor import TextProcessor
from .feature_extractor import FeatureExtractor


class SpamDetector:
    """Main spam detection classifier"""
    
    def __init__(self, algorithm='naive_bayes'):
        """
        Initialize spam detector
        
        Args:
            algorithm (str): 'naive_bayes', 'svm', or 'logistic_regression'
        """
        self.algorithm = algorithm
        self.model = None
        self.text_processor = TextProcessor()
        self.feature_extractor = FeatureExtractor(method='tfidf')
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the selected ML model"""
        if self.algorithm == 'naive_bayes':
            self.model = MultinomialNB()
        elif self.algorithm == 'svm':
            self.model = LinearSVC(random_state=42, max_iter=2000)
        elif self.algorithm == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def train(self, X_train, y_train):
        """
        Train the spam detector
        
        Args:
            X_train (list): Training texts
            y_train (array): Training labels
            
        Returns:
            self: For method chaining
        """
        print(f"Training {self.algorithm} model...")
        
        # Process texts
        X_processed = self.text_processor.batch_process(X_train)
        
        # Extract features
        X_features = self.feature_extractor.fit_transform(X_processed)
        
        # Train model
        self.model.fit(X_features, y_train)
        
        print(f"Model trained successfully!")
        return self
    
    def predict(self, text):
        """
        Predict if text is spam
        
        Args:
            text (str): Email text
            
        Returns:
            int: 1 for spam, 0 for ham
        """
        if isinstance(text, list):
            return self.predict_batch(text)
        
        # Process text
        processed_text = self.text_processor.process(text)
        
        # Extract features
        features = self.feature_extractor.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(features)[0]
        return prediction
    
    def predict_batch(self, texts):
        """
        Predict for multiple texts
        
        Args:
            texts (list): List of email texts
            
        Returns:
            array: Predictions
        """
        # Process texts
        X_processed = self.text_processor.batch_process(texts)
        
        # Extract features
        X_features = self.feature_extractor.transform(X_processed)
        
        # Predict
        predictions = self.model.predict(X_features)
        return predictions
    
    def predict_proba(self, text):
        """
        Get probability estimates
        
        Args:
            text (str): Email text
            
        Returns:
            float: Probability of spam
        """
        # Process text
        processed_text = self.text_processor.process(text)
        
        # Extract features
        features = self.feature_extractor.transform([processed_text])
        
        # Get probability
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features)[0]
            return proba[1]  # Probability of spam class
        else:
            # For SVM, use decision function
            decision = self.model.decision_function(features)[0]
            return 1 / (1 + np.exp(-decision))  # Sigmoid transformation
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (list): Test texts
            y_test (array): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        # Process texts
        X_processed = self.text_processor.batch_process(X_test)
        
        # Extract features
        X_features = self.feature_extractor.transform(X_processed)
        
        # Predictions
        y_pred = self.model.predict(X_features)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        # Print results
        print(f"\n{'='*50}")
        print(f"Model Evaluation - {self.algorithm.upper()}")
        print(f"{'='*50}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print(f"\nClassification Report:")
        print(metrics['classification_report'])
        
        return metrics
    
    def save_model(self, filepath):
        """
        Save model to file
        
        Args:
            filepath (str): Path to save model
        """
        model_data = {
            'model': self.model,
            'algorithm': self.algorithm,
            'vectorizer': self.feature_extractor.vectorizer,
            'text_processor': self.text_processor
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load model from file
        
        Args:
            filepath (str): Path to model file
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.algorithm = model_data['algorithm']
        self.feature_extractor.vectorizer = model_data['vectorizer']
        self.text_processor = model_data['text_processor']
        
        print(f"Model loaded from {filepath}")
