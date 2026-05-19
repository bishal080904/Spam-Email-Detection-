"""Spam Email Detection Package"""

__version__ = "1.0.0"
__author__ = "Bishal"

from .classifier import SpamDetector
from .text_processor import TextProcessor
from .feature_extractor import FeatureExtractor
from .data_loader import DataLoader

__all__ = [
    'SpamDetector',
    'TextProcessor',
    'FeatureExtractor',
    'DataLoader'
]
