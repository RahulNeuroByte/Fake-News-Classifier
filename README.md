# 📰 Fake News Classification Project

A comprehensive machine learning project for detecting fake news using multiple algorithms and a user-friendly Streamlit web application.

## 🚀 Live Demo

[Click here to view the deployed Fake News Classifier App](https://fake-news-classifier-rahulneurobyte.streamlit.app/)

## 🎯 Project Overview

This project implements a complete fake news classification system with the following features:

- **Multiple ML Models**: Naive Bayes, Logistic Regression, KNN, SVM, Random Forest
- **Text Vectorization**: Both Count Vectorizer and TF-IDF Vectorizer
- **Hyperparameter Tuning**: Automated optimization for best performance
- **Interactive Web App**: Streamlit-based interface for real-time predictions
- **Batch Processing**: Support for CSV file uploads and bulk predictions
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations
- **Easy Deployment**: Ready for local and cloud deployment

## 1 📁 Project Structure

```
Fake_News_Classifier/
│
├── data/                          # Dataset storage
│   └── dataset.csv               # Main dataset (place your dataset here)
│
├── models/                       # Trained models and vectorizers
│   ├── *.pkl                    # Saved model files
│   ├── *_vectorizer.pkl         # Vectorizer files
│   ├── best_model.pkl           # Best performing model
│   └── best_model_info.pkl      # Model metadata
│
├── results/                      # Evaluation results and visualizations
│   ├── *.png                    # Confusion matrices and plots
│   ├── model_comparison.csv     # Model performance comparison
│   └── evaluation_report.md     # Detailed evaluation report
│
├── app/                          # Streamlit web application
│   └── streamlit_app.py         # Main application file
│
├── preprocessing.py              # Text preprocessing utilities
├── model_training.py            # Model training pipeline
├── model_evaluation.py          # Model evaluation utilities
├── model_tuning.py              # Hyperparameter tuning
├── prediction_pipeline.py       # Prediction pipeline
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```



```
Fake_News_Classifier/
│
├── data/ # Raw dataset
│ └── dataset.csv
│
├── models/ # Trained models & vectorizers
│ ├── best_model.pkl
│ ├── vectorizer.pkl
│
├── results/ # Model evaluation results
│ ├── confusion_matrix.png
│ └── model_comparison.csv
│
├── app/ # Streamlit application
│ └── app.py
│
├── preprocessing.py # Text preprocessing functions
├── model_training.py # Model training script
├── model_evaluation.py # Evaluation script
├── model_tuning.py # Hyperparameter tuning
├── prediction_pipeline.py # Prediction logic
├── requirements.txt # Python dependencies
└── README.md # You're here!

```
---

## 📊 Features

### ✅ ML Models Implemented
- Logistic Regression
- Multinomial Naive Bayes
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest

### 🔤 Vectorization Methods
- CountVectorizer
- TF-IDF Vectorizer

### 🧪 Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- ROC Curve & AUC

### 🌐 Streamlit Web App
- Real-time prediction
- Batch prediction (CSV upload)
- Model performance view
- Probability & confidence visualization

---

## ⚙️ Setup & Usage

### 1. 🧾 Data Preparation

Ensure your dataset follows this format:

- CSV format placed in `data/dataset.csv`
- Must have:
  - A text column: `text`, `news`, or `content`
  - A label column: `label`, `target`, or `class`
  - Labels: `0` = Fake, `1` = Real

---

### 2. 🏗️ Model Training

Run:

```bash
python model_training.py

✅ Conclusion
This Fake News Classifier project is a complete end-to-end system built with real-world practicality in mind. It combines robust machine learning models, clean text processing, and an interactive Streamlit interface — making it a great tool for both educational purposes and potential deployment.

By allowing both individual and bulk predictions with live confidence visualization, the system serves as a solid foundation for combating misinformation online.

🔮 Future Enhancements:

Deep learning models (BERT, LSTM)

Real-time news scraping & classification

Multi-language support

Browser extension integration

Happy Classifying! 📰🚀

