"""Main execution script for spam email detection"""

import argparse
import sys
from src.data_loader import DataLoader
from src.classifier import SpamDetector


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Spam Email Detection System')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--predict', type=str, help='Predict for a given email text')
    parser.add_argument('--algorithm', type=str, default='naive_bayes',
                       choices=['naive_bayes', 'svm', 'logistic_regression'],
                       help='Algorithm to use')
    parser.add_argument('--data', type=str, default='data/emails.csv',
                       help='Path to data file')
    parser.add_argument('--model', type=str, default='models/trained_model.pkl',
                       help='Path to model file')
    
    args = parser.parse_args()
    
    # Train mode
    if args.train:
        print("\n" + "="*60)
        print("TRAINING SPAM DETECTION MODEL")
        print("="*60)
        
        # Load data
        print("\nLoading data...")
        loader = DataLoader(test_size=0.2)
        try:
            X_train, X_test, y_train, y_test = loader.load_and_preprocess(args.data)
        except Exception as e:
            print(f"Error loading data: {e}")
            print(f"Please ensure {args.data} exists with 'text' and 'label' columns")
            return
        
        # Train model
        print(f"\nTraining {args.algorithm} model...")
        detector = SpamDetector(algorithm=args.algorithm)
        detector.train(X_train, y_train)
        
        # Evaluate
        print("\nEvaluating model...")
        detector.evaluate(X_test, y_test)
        
        # Save model
        print(f"\nSaving model to {args.model}...")
        detector.save_model(args.model)
        
        print("\nTraining completed!")
    
    # Evaluate mode
    elif args.evaluate:
        print("\n" + "="*60)
        print("EVALUATING SPAM DETECTION MODEL")
        print("="*60)
        
        try:
            # Load data
            print("\nLoading data...")
            loader = DataLoader(test_size=0.2)
            X_train, X_test, y_train, y_test = loader.load_and_preprocess(args.data)
            
            # Load model
            print(f"\nLoading model from {args.model}...")
            detector = SpamDetector()
            detector.load_model(args.model)
            
            # Evaluate
            print("\nEvaluating model...")
            detector.evaluate(X_test, y_test)
        
        except Exception as e:
            print(f"Error: {e}")
            print(f"Make sure {args.model} exists. Train the model first with --train")
    
    # Predict mode
    elif args.predict:
        print("\n" + "="*60)
        print("SPAM EMAIL PREDICTION")
        print("="*60)
        
        try:
            # Load model
            print(f"\nLoading model from {args.model}...")
            detector = SpamDetector()
            detector.load_model(args.model)
            
            # Make prediction
            print(f"\nAnalyzing: {args.predict[:100]}...")
            prediction = detector.predict(args.predict)
            probability = detector.predict_proba(args.predict)
            
            result = "SPAM" if prediction == 1 else "HAM (Legitimate)"
            print(f"\nPrediction: {result}")
            print(f"Confidence: {probability:.2%}")
        
        except Exception as e:
            print(f"Error: {e}")
            print(f"Make sure {args.model} exists. Train the model first with --train")
    
    # No arguments
    else:
        # Demo mode
        print("\n" + "="*60)
        print("SPAM EMAIL DETECTION SYSTEM - DEMO")
        print("="*60)
        print("\nUsage:")
        print("  python main.py --train              # Train the model")
        print("  python main.py --evaluate           # Evaluate the model")
        print("  python main.py --predict 'text'     # Predict for text")
        print("\nExample:")
        print("  python main.py --predict 'Click here to win free money!'")
        
        # Try to demonstrate with example
        print("\n" + "-"*60)
        print("DEMO: Testing with sample emails...")
        print("-"*60)
        
        try:
            detector = SpamDetector(algorithm='naive_bayes')
            detector.load_model(args.model)
            
            test_emails = [
                "Click here to win free money now!",
                "Dear colleague, please review the attached proposal.",
                "Congratulations! You've won a prize!",
                "Meeting scheduled for tomorrow at 2 PM."
            ]
            
            for email in test_emails:
                prediction = detector.predict(email)
                probability = detector.predict_proba(email)
                result = "SPAM" if prediction == 1 else "HAM"
                print(f"\n[{result}] {email}")
                print(f"  Confidence: {probability:.2%}")
        
        except:
            print("\nModel not found. Train first: python main.py --train")


if __name__ == '__main__':
    main()
