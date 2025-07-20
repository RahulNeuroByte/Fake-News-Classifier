# Fake News Classification Model Evaluation Report

Generated on: 2025-07-20 18:39:19

## Model Performance Summary

### Top Performing Models

**10. svm_tfidf**
   - Accuracy: 0.9493
   - Precision: 0.9493
   - Recall: 0.9493
   - F1-Score: 0.9493

**3. logistic_regression_count**
   - Accuracy: 0.9409
   - Precision: 0.9409
   - Recall: 0.9409
   - F1-Score: 0.9409
   - AUC Score: 0.9816

**4. logistic_regression_tfidf**
   - Accuracy: 0.9385
   - Precision: 0.9386
   - Recall: 0.9385
   - F1-Score: 0.9385
   - AUC Score: 0.9871

### Detailed Metrics Comparison

| model                     |   accuracy |   precision |   recall |   f1_score |   auc_score |
|:--------------------------|-----------:|------------:|---------:|-----------:|------------:|
| svm_tfidf                 |     0.9493 |      0.9493 |   0.9493 |     0.9493 |    nan      |
| logistic_regression_count |     0.9409 |      0.9409 |   0.9409 |     0.9409 |      0.9816 |
| logistic_regression_tfidf |     0.9385 |      0.9386 |   0.9385 |     0.9385 |      0.9871 |
| random_forest_count       |     0.9380 |      0.9383 |   0.9380 |     0.9380 |      0.9844 |
| random_forest_tfidf       |     0.9361 |      0.9361 |   0.9361 |     0.9361 |      0.9848 |
| svm_count                 |     0.9315 |      0.9315 |   0.9315 |     0.9315 |    nan      |
| naive_bayes_tfidf         |     0.8892 |      0.8913 |   0.8892 |     0.8890 |      0.9660 |
| naive_bayes_count         |     0.8793 |      0.8809 |   0.8793 |     0.8792 |      0.9486 |
| knn_count                 |     0.7901 |      0.8258 |   0.7901 |     0.7842 |      0.8665 |
| knn_tfidf                 |     0.6459 |      0.7619 |   0.6459 |     0.6015 |      0.7166 |


### Key Insights

- **Best Overall Model**: svm_tfidf with 0.9493 accuracy
- **Best AUC Score**: logistic_regression_tfidf with 0.9871 AUC
- **Better Vectorizer**: Count Vectorizer (Avg Accuracy: 0.8960)

### Recommendations

- Use the best performing model for production deployment
- Consider ensemble methods if multiple models perform similarly
- Monitor model performance on new data and retrain if necessary