# 📰 Fake News Classification Project

A comprehensive machine learning project for detecting fake news using multiple algorithms and a user-friendly Streamlit web application.

## 🎯 Project Overview

This project implements a complete fake news classification system with the following features:

- **Multiple ML Models**: Naive Bayes, Logistic Regression, KNN, SVM, Random Forest
- **Text Vectorization**: Both Count Vectorizer and TF-IDF Vectorizer
- **Hyperparameter Tuning**: Automated optimization for best performance
- **Interactive Web App**: Streamlit-based interface for real-time predictions
- **Batch Processing**: Support for CSV file uploads and bulk predictions
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations
- **Easy Deployment**: Ready for local and cloud deployment

## 📁 Project Structure

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

## 🚀 Quick Start Guide

### 1. Environment Setup (Windows)



### 2. Data Preparation

1. **Place your dataset** in the `data/` folder as `dataset.csv`
2. **Dataset format**: Your CSV should have:
   - A text column (named 'text', 'news', or 'content')
   - A label column (named 'label', 'target', or 'class')
   - Labels should be binary: 0 for Fake, 1 for Real

### 3. Model Training (Windows)



This will:
- Preprocess the text data
- Train multiple models with different vectorizers
- Evaluate and save all models
- Generate performance visualizations
- Save the best performing model

### 4. Hyperparameter Tuning (Optional - Windows)


### 5. Launch the Web Application (Windows)


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

2. **Batch File Processing**
   - Upload CSV files for bulk predictions
   - Downloadable results
   - Summary statistics and visualizations

3. **Model Performance Dashboard**
   - Model comparison charts
   - Confusion matrices
   - Performance metrics

### Model Training Features

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

## 🌐 Deployment Options

### Local Deployment (Windows)



### Streamlit Cloud Deployment

1. Push your project to GitHub
2. Connect to [[Streamlit Cloud](https://streamlit.io/cloud](https://fake-news-classifier-rahulneurobyte.streamlit.app/))
3. Deploy directly from your repository



## 🔍 Usage Examples

### Single Prediction

```python
from prediction_pipeline import FakeNewsPredictionPipeline

# Initialize pipeline
pipeline = FakeNewsPredictionPipeline()
pipeline.load_best_model()

# Make prediction
text = "Your news article text here..."
result = pipeline.predict_single_text(text)

print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Batch Prediction

```python
# Predict from CSV file
results_df = pipeline.predict_from_file('your_file.csv', 'text_column')
print(results_df.head())
```


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

## 🛠️ Troubleshooting

### Common Issues

1. **"Model files not found"**
   - Run `python model_training.py` first
   - Check that `models/` directory contains `.pkl` files

2. **"Dataset not found"**
   - Ensure `dataset.csv` is in the `data/` folder
   - Check file path in training scripts

3. **"Import errors"**
   - Install all requirements: `pip install -r requirements.txt`
   - Download NLTK data as shown in setup

4. **"Memory errors during training"**
   - Reduce `max_features` in vectorizers
   - Use smaller dataset for testing
   - Consider using incremental learning

### Performance Issues

1. **Slow predictions**
   - Use lighter models (Naive Bayes, Logistic Regression)
   - Reduce vectorizer features
   - Cache models in Streamlit app

2. **Low accuracy**
   - Run hyperparameter tuning
   - Check data quality and preprocessing
   - Try ensemble methods

## 📝 Development Notes

### Adding New Features

1. **New Models**: Add to `model_training.py` and `model_tuning.py`
2. **New Metrics**: Extend `model_evaluation.py`
3. **UI Improvements**: Modify `app/streamlit_app.py`
4. **New Visualizations**: Add to evaluation or app modules

### Code Organization

- **Modular design**: Each component is in separate files
- **Clear file paths**: All paths marked with comments
- **Error handling**: Comprehensive error checking
- **Documentation**: Inline comments and docstrings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Scikit-learn** for machine learning algorithms
- **Streamlit** for the web application framework
- **NLTK** for natural language processing
- **Plotly** for interactive visualizations

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the code comments for file path guidance
3. Ensure all dependencies are installed correctly
4. Verify dataset format and location

---

**Happy Fake News Detection! 🎯**

