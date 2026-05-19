# Spam Email Detection System

A machine learning-based spam email detection system built with Python. This project uses natural language processing and classification algorithms to identify and filter spam emails.

## Features

- **Email Classification**: Binary classification (spam/ham) using multiple ML algorithms
- **NLP Processing**: Text preprocessing, tokenization, and feature extraction
- **Multiple Algorithms**: Naive Bayes, SVM, and Logistic Regression implementations
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, and F1-score
- **Easy Integration**: Simple API for classifying new emails

## Project Structure

```
Spam-Email-Detection-/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── data/
│   ├── spam.csv             # Sample spam emails dataset
│   └── ham.csv              # Sample legitimate emails dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── text_processor.py    # NLP text processing
│   ├── feature_extractor.py # Feature extraction methods
│   └── classifier.py        # ML classification models
├── models/
│   └── trained_model.pkl    # Serialized trained model
├── notebooks/
│   └── spam_detection.ipynb # Jupyter notebook with analysis
└── main.py                  # Main execution script
```

## Installation

### Requirements
- Python 3.7+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/bishal080904/Spam-Email-Detection-.git
cd Spam-Email-Detection-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python main.py --train
```

### Classifying an Email

```python
from src.classifier import SpamDetector

# Initialize detector
detector = SpamDetector()
detector.load_model('models/trained_model.pkl')

# Classify an email
email_text = "Click here to win free money!"
prediction = detector.predict(email_text)

if prediction == 1:
    print("SPAM")
else:
    print("HAM (Legitimate)")
```

### Evaluate Model

```bash
python main.py --evaluate
```

## Algorithm Details

### 1. Naive Bayes Classifier
- **Pros**: Fast, works well with text data, good baseline
- **Cons**: Assumes feature independence

### 2. Support Vector Machine (SVM)
- **Pros**: Effective in high-dimensional spaces, good generalization
- **Cons**: Slower training, memory intensive

### 3. Logistic Regression
- **Pros**: Interpretable, efficient, probabilistic predictions
- **Cons**: Linear decision boundary

## Performance Metrics

The system is evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / All positives
- **Recall**: True positives / All actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis

## Dataset

The project uses:
- **Spam dataset**: Common spam email patterns and phishing attempts
- **Ham dataset**: Legitimate business and personal emails
- **Preprocessing**: Lowercasing, punctuation removal, tokenization, stop word removal

## Future Enhancements

- [ ] Add deep learning models (LSTM, CNN)
- [ ] Implement email header analysis
- [ ] Add attachment scanning
- [ ] Create REST API endpoint
- [ ] Build web interface
- [ ] Add real-time monitoring
- [ ] Implement active learning

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## License

MIT License - See LICENSE file for details

## Contact

**Author**: Bishal  
**GitHub**: [bishal080904](https://github.com/bishal080904)  
**Email**: Contact via GitHub profile

## Disclaimer

This system is for educational purposes. Accuracy depends on training data quality and model selection. Always combine with other security measures for production use.
