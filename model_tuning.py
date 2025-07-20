

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from preprocessing import preprocess_text
#from config.path_config import MODEL_DIR, RESULT_DIR

import warnings
warnings.filterwarnings("ignore")

class HyperparameterTuner:
    def __init__(self, data_path="data/dataset.csv"): # FILE PATH: 
        """
        Initialize the hyperparameter tuner
        
        Args:
            data_path (str): Path to the dataset CSV file
        """
        self.data_path = data_path
        self.best_models = {}
        self.best_params = {}
        self.best_scores = {}

    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("Loading dataset for tuning...")
        # FILE PATH: Make sure the dataset.csv is in the data/ folder
        self.df = pd.read_csv(self.data_path)

        # Handle different possible column names
        if 'text' in self.df.columns:
            text_col = 'text'
        elif 'news' in self.df.columns:
            text_col = 'news'
        elif 'content' in self.df.columns:
            text_col = 'content'
        else:
            text_col = self.df.columns[0]  # Assume first column is text
            
        if 'label' in self.df.columns:
            label_col = 'label'
        elif 'target' in self.df.columns:
            label_col = 'target'
        elif 'class' in self.df.columns:
            label_col = 'class'
        else:
            label_col = self.df.columns[-1]  # Assume last column is label

        self.df['processed_text'] = self.df[text_col].astype(str).apply(preprocess_text)
        self.X = self.df['processed_text']
        self.y = self.df[label_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        print("Data loaded and split for tuning.")

    def tune_model(self, model_name, model, param_grid, vectorizer_type='tfidf'):
        """Tune a single model using GridSearchCV"""
        print(f"\nStarting hyperparameter tuning for {model_name} with {vectorizer_type}...")

        if vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        elif vectorizer_type == 'count':
            vectorizer = CountVectorizer(max_features=5000, stop_words='english')
        else:
            raise ValueError("vectorizer_type must be 'tfidf' or 'count'")

        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', model)
        ])

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)

        self.best_models[f'{model_name}_{vectorizer_type}'] = grid_search.best_estimator_
        self.best_params[f'{model_name}_{vectorizer_type}'] = grid_search.best_params_
        self.best_scores[f'{model_name}_{vectorizer_type}'] = grid_search.best_score_

        print(f"Best parameters for {model_name} ({vectorizer_type}): {grid_search.best_params_}")
        print(f"Best accuracy for {model_name} ({vectorizer_type}): {grid_search.best_score_:.4f}")

        # Save the best model
        # FILE PATH: Tuned models will be saved in models/ folder
        os.makedirs('models', exist_ok=True)
        with open(f'models/tuned_{model_name}_{vectorizer_type}.pkl', 'wb') as f:
            pickle.dump(grid_search.best_estimator_, f)
        print(f"Tuned {model_name} ({vectorizer_type}) model saved.")

    def run_tuning_pipeline(self):
        """Run tuning for selected models"""
        self.load_and_preprocess_data()

        # Define parameter grids for tuning
        param_grids = {
            'logistic_regression': {
                'classifier__C': [0.1, 1, 10],
                'classifier__solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20]
            },
            # Add more models and their parameter grids as needed
            'naive_bayes': {
                'classifier__alpha': [0.1, 0.5, 1.0]
            },
            
          
        }

        models_to_tune = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42),
            'naive_bayes': MultinomialNB(),
        
           
        }

        for model_name, model_instance in models_to_tune.items():
            if model_name in param_grids:
                self.tune_model(model_name, model_instance, param_grids[model_name], vectorizer_type='tfidf')
                self.tune_model(model_name, model_instance, param_grids[model_name], vectorizer_type='count')

        print("\n--- Tuning Summary ---")
        for model_name, score in self.best_scores.items():
            print(f"Model: {model_name}, Best Score: {score:.4f}")
            print(f"Best Params: {self.best_params[model_name]}")

        # Save best overall model info
        best_overall_model_name = max(self.best_scores, key=self.best_scores.get)
        best_overall_score = self.best_scores[best_overall_model_name]
        best_overall_params = self.best_params[best_overall_model_name]

        model_info = {
            'model_name': best_overall_model_name,
            'accuracy': best_overall_score,
            'params': best_overall_params
        }
        # FILE PATH: Best overall tuned model info saved in models/ folder
        with open('models/best_tuned_model_info.pkl', 'wb') as f:
            pickle.dump(model_info, f)
        print(f"\nBest overall tuned model: {best_overall_model_name} with accuracy {best_overall_score:.4f}")

if __name__ == '__main__':
    tuner = HyperparameterTuner()
    tuner.run_tuning_pipeline()


