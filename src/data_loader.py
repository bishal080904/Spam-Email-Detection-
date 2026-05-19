"""Data loading and preprocessing module"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DataLoader:
    """Load and preprocess email datasets"""
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize DataLoader
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
    
    def load_data(self, filepath):
        """
        Load email data from CSV file
        
        Args:
            filepath (str): Path to CSV file
            
        Returns:
            pandas.DataFrame: Loaded data
        """
        try:
            data = pd.read_csv(filepath)
            print(f"Data loaded successfully from {filepath}")
            print(f"Shape: {data.shape}")
            return data
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return None
    
    def preprocess_data(self, data, email_column='text', label_column='label'):
        """
        Preprocess email data
        
        Args:
            data (pd.DataFrame): Raw data
            email_column (str): Name of email text column
            label_column (str): Name of label column
            
        Returns:
            tuple: (X, y) - Texts and labels
        """
        # Remove missing values
        data = data.dropna(subset=[email_column, label_column])
        
        X = data[email_column].values
        y = data[label_column].values
        
        # Encode labels: spam=1, ham=0
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Data preprocessed: {len(X)} samples")
        print(f"Class distribution: {np.bincount(y_encoded)}")
        
        return X, y_encoded
    
    def split_data(self, X, y):
        """
        Split data into train and test sets
        
        Args:
            X (array): Feature data
            y (array): Labels
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def load_and_preprocess(self, filepath, email_column='text', label_column='label'):
        """
        Load and preprocess data in one step
        
        Args:
            filepath (str): Path to CSV file
            email_column (str): Email text column name
            label_column (str): Label column name
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        data = self.load_data(filepath)
        if data is None:
            return None
        
        X, y = self.preprocess_data(data, email_column, label_column)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        return X_train, X_test, y_train, y_test
