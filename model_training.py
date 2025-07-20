

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import preprocess_text
#from config.path_config import MODEL_DIR, RESULT_DIR

import warnings
warnings.filterwarnings('ignore')

class FakeNewsModelTrainer:
    def __init__(self, data_path='data/dataset.csv'):  # FILE PATH
        """
        Initialize the model trainer
        
        Args:
            data_path (str): Path to the dataset CSV file
        """
        self.data_path = data_path
        self.models = {}
        self.vectorizers = {}
        self.results = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("Loading dataset...")
        # FILE PATH: Make sure the dataset.csv is in the data/ folder
        self.df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        
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
            
        print(f"Using text column: {text_col}")
        print(f"Using label column: {label_col}")
        
        # Preprocess text
        print("Preprocessing text...")
        self.df['processed_text'] = self.df[text_col].astype(str).apply(preprocess_text)
        
        # Prepare features and target
        self.X = self.df['processed_text']
        self.y = self.df[label_col]
        
        print(f"Label distribution:\n{self.y.value_counts()}")
        
    def create_vectorizers(self):
        """Create and fit vectorizers"""
        print("Creating vectorizers...")
        
        # Count Vectorizer
        self.count_vectorizer = CountVectorizer(max_features=5000, stop_words='english')
        self.X_count = self.count_vectorizer.fit_transform(self.X)
        
        # TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.X_tfidf = self.tfidf_vectorizer.fit_transform(self.X)
        
        # Save vectorizers-------------------------



        # FILE PATH: Vectorizers will be saved in models/ folder
        os.makedirs('models', exist_ok=True)
        with open('models/count_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.count_vectorizer, f)
        with open('models/tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
            
        print("Vectorizers created and saved!")
        
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print("Splitting data...")
        
        # Split for Count Vectorizer
        self.X_count_train, self.X_count_test, self.y_train, self.y_test = train_test_split(
            self.X_count, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Split for TF-IDF Vectorizer
        self.X_tfidf_train, self.X_tfidf_test, _, _ = train_test_split(
            self.X_tfidf, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"Training set size: {self.X_count_train.shape[0]}")
        print(f"Test set size: {self.X_count_test.shape[0]}")
        
    def train_models(self):
        """Train all models with both vectorizers"""
        print("Training models...")
        
        # Define models
        models_config = {
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            #'knn': KNeighborsClassifier(n_neighbors=5),
            #'svm': SVC(random_state=42, probability=True),
            #'svm' : SVC(kernel='linear', max_iter=10000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        # Train models with both vectorizers
        for model_name, model in models_config.items():
            print(f"Training {model_name}...")
            
            # Train with Count Vectorizer
            model_count = model.__class__(**model.get_params())
            model_count.fit(self.X_count_train, self.y_train)
            
            # Train with TF-IDF Vectorizer
            model_tfidf = model.__class__(**model.get_params())
            model_tfidf.fit(self.X_tfidf_train, self.y_train)
            
            # Store models
            self.models[f'{model_name}_count'] = model_count
            self.models[f'{model_name}_tfidf'] = model_tfidf
            
            # Save models-------------------










            # FILE PATH: Models will be saved in models/ folder
            with open(f'models/{model_name}_count.pkl', 'wb') as f:
                pickle.dump(model_count, f)
            with open(f'models/{model_name}_tfidf.pkl', 'wb') as f:
                pickle.dump(model_tfidf, f)
                
        print("All models trained and saved!")
        
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("Evaluating models...")
        
        # Create results directory
        # FILE PATH: Results will be saved in results/ folder
        os.makedirs('results', exist_ok=True)
        
        results_data = []
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            
            # Choose appropriate test data
            if 'count' in model_name:
                X_test = self.X_count_test
            else:
                X_test = self.X_tfidf_test
                
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Store results
            self.results[model_name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': classification_report(self.y_test, y_pred),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            
            results_data.append({
                'Model': model_name,
                'Accuracy': accuracy
            })
            
            # Save confusion matrix plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_matrix(self.y_test, y_pred), annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            # FILE PATH: Confusion matrix plots will be saved in results/ folder
            plt.savefig(f'results/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        # Create results summary
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        # FILE PATH: Results summary will be saved in results/ folder
        results_df.to_csv('results/model_comparison.csv', index=False)
        
        # Plot model comparison
        plt.figure(figsize=(12, 8))
        sns.barplot(data=results_df, x='Accuracy', y='Model', palette='viridis')
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Accuracy')
        for i, v in enumerate(results_df['Accuracy']):
            plt.text(v + 0.001, i, f'{v:.4f}', va='center')
        # FILE PATH: Model comparison plot will be saved in results/ folder
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Model evaluation completed!")
        print("\nTop 3 Models:")
        print(results_df.head(3).to_string(index=False))
        
        return results_df
        
    def save_best_model(self):
        """Save the best performing model"""
        # Find best model
        best_accuracy = 0
        best_model_name = None
        
        for model_name, results in self.results.items():
            if results['accuracy'] > best_accuracy:
                best_accuracy = results['accuracy']
                best_model_name = model_name
                
        print(f"Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")
        
        # Copy best model
        best_model = self.models[best_model_name]
        # FILE PATH: Best model will be saved in models/ folder
        with open('models/best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
            
        # Save model info
        model_info = {
            'model_name': best_model_name,
            'accuracy': best_accuracy,
            'vectorizer_type': 'count' if 'count' in best_model_name else 'tfidf'
        }
        
        with open('models/best_model_info.pkl', 'wb') as f:
            pickle.dump(model_info, f)
            
        print("Best model saved!")
        
    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        print("Starting full training pipeline...")
        
        self.load_and_preprocess_data()
        self.create_vectorizers()
        self.split_data()
        self.train_models()
        results_df = self.evaluate_models()
        self.save_best_model()
        
        print("Training pipeline completed successfully!")
        return results_df

if __name__ == '__main__':
    # FILE PATH: Make sure dataset.csv is in data/ folder before running
    trainer = FakeNewsModelTrainer()
    results = trainer.run_full_pipeline()


