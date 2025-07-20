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



### 2. Data Preparation

1. **Place your dataset** in the `data/` folder as `dataset.csv`
2. **Dataset format**: Your CSV should have:
   - A text column (named 'text', 'news', or 'content')
   - A label column (named 'label', 'target', or 'class')
   - Labels should be binary: 0 for Fake, 1 for Real

### 3. Model Training

This will:
- Preprocess the text data
- Train multiple models with different vectorizers
- Evaluate and save all models
- Generate performance visualizations
- Save the best performing model

### 4. Hyperparameter Tuning
- Tuning various model for high accuracy 

### 5. Launch the Web Application 

# Run the Streamlit app
## Visit this link (https://fake-news-classifier-rahulneurobyte.streamlit.app/)


## 🔧 Configuration and Customization

### File Paths Configuration

All file paths are clearly marked with `# FILE PATH:` comments. Key locations to update if needed:

1. **Dataset location**: Update `data_path` in training scripts
2. **Models directory**: Update `models_dir` in prediction pipeline
3. **Results directory**: Update `results_dir` in evaluation scripts
4. **App imports**: Update `sys.path.append()` in Streamlit app

### Model Configuration

To add new models or modify existing ones:

1. **Edit `model_training.py`**: Add new models to `models_config` dictionary
2. **Edit `model_tuning.py`**: Add parameter grids for new models
3. **Update preprocessing**: Modify `preprocessing.py` for custom text processing

### Vectorizer Configuration

Current vectorizers use:
- `max_features=5000`: Adjust based on dataset size
- `stop_words='english'`: Change for other languages
- Custom preprocessing pipeline in `preprocessing.py`

## 📊 Features and Capabilities

### Web Application Features

1. **Single Text Prediction**
   - Real-time classification of news articles
   - Confidence scores and probability breakdown
   - Interactive visualizations

1. **Multiple Algorithms**
   - Naive Bayes (MultinomialNB)
   - Logistic Regression
   - K-Nearest Neighbors
   - Support Vector Machine
   - Random Forest

2. **Text Vectorization**
   - Count Vectorizer
   - TF-IDF Vectorizer
   - Automatic comparison and selection

3. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrices
   - ROC curves and AUC scores
   - Detailed classification reports

## 📈 Performance Optimization

### For Large Datasets

1. **Increase max_features** in vectorizers
2. **Use incremental learning** for very large datasets
3. **Consider feature selection** techniques
4. **Implement data sampling** for faster training

### For Better Accuracy

1. **Run hyperparameter tuning**: `python model_tuning.py`
2. **Experiment with ensemble methods**
3. **Try different preprocessing techniques**
4. **Use domain-specific stop words**

---

**Happy Fake News Detection! 🎯**
















🔧 Setup Instructions & Usage Guide
1. 📁 Data Preparation
Prepare your dataset before training:

Place your dataset in the data/ directory as dataset.csv

Dataset format must include:

A text column named either text, news, or content

A label column named either label, target, or class

Binary labels:

0 → Fake news

1 → Real news

2. ⚙️ Model Training
Running model_training.py will:

Preprocess text data (cleaning, lemmatization, etc.)

Train multiple machine learning models using:

Count Vectorizer

TF-IDF Vectorizer

Evaluate all models

Save trained models and vectorizers

Generate performance visualizations

Automatically save the best-performing model

3. 🎯 Hyperparameter Tuning (Optional)
For improved accuracy:

Run model_tuning.py

Performs GridSearchCV or RandomizedSearchCV

Fine-tunes hyperparameters for each model

4. 🚀 Launch the Streamlit Web App
Once training is complete:

bash
Copy
Edit
streamlit run app/app.py
Or visit the live version:

👉 Live App

⚙️ Configuration & Customization
🔄 File Paths Configuration
Key configurable file paths (marked as # FILE PATH: in scripts):

Component	File Path to Update
Dataset	data_path in training scripts
Models directory	models_dir in prediction_pipeline.py
Results directory	results_dir in evaluation scripts
App imports	sys.path.append() in app.py

🧠 Model & Preprocessing Customization
Add New Models:

Update models_config in model_training.py

Add New Hyperparameters:

Edit grid in model_tuning.py

Customize Preprocessing:

Modify preprocessing.py for cleaning, stopword filtering, etc.

🧪 Vectorizer Configuration
Default vectorizers:

CountVectorizer

TfidfVectorizer

Settings:

max_features=5000 — adjust for large datasets

stop_words='english' — can be customized

Custom preprocessing pipeline via preprocessing.py

📊 App Features & Capabilities
🔍 Web Application Modes
Single Text Prediction

Real-time classification of a headline/article

Displays:

Predicted label (Fake/Real)

Confidence score

Probability distribution

Visuals (bar chart & gauge)

Batch File Prediction

Upload .csv file

Returns predictions for all rows

Summary stats and download option

Model Performance View

Accuracy comparison of all models

Visualizations (confusion matrices, bar charts)

🧠 Supported ML Models
Logistic Regression

Multinomial Naive Bayes

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Random Forest Classifier

📈 Evaluation Metrics
Accuracy, Precision, Recall, F1-Score

Confusion Matrices

ROC Curve & AUC Score

Classification Reports

⚡ Performance Optimization Tips
For Large Datasets
Increase max_features in vectorizers

Use incremental learning algorithms (e.g., partial_fit)

Implement feature selection

Use stratified sampling to balance training

For Higher Accuracy
Run hyperparameter tuning (model_tuning.py)

Use ensemble models (Voting, Stacking)

Improve text cleaning (domain-specific filters)

Include metadata features (e.g., source credibility)

✅ Conclusion
This Fake News Classifier project is a complete end-to-end system built with real-world practicality in mind. It combines robust machine learning models, clean text processing, and an interactive Streamlit interface — making it a great tool for both educational purposes and potential deployment.

By allowing both individual and bulk predictions with live confidence visualization, the system serves as a solid foundation for combating misinformation online.

🔮 Future Enhancements:

Deep learning models (BERT, LSTM)

Real-time news scraping & classification

Multi-language support

Browser extension integration

Happy Classifying! 📰🚀

