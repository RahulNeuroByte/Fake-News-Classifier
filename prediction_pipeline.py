
import pickle
import pandas as pd
import numpy as np
import os
from utils import lemmatization_sentence as preprocess_text

import warnings
warnings.filterwarnings('ignore')

class FakeNewsPredictionPipeline:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.model = None
        self.vectorizer = None
        self.model_info = None

    def load_best_model(self):
        try:
            with open(os.path.join(self.models_dir, 'best_model_info.pkl'), 'rb') as f:
                self.model_info = pickle.load(f)

            model_name = self.model_info['model_name']
            vectorizer_type = self.model_info['vectorizer_type']

            with open(os.path.join(self.models_dir, 'best_model.pkl'), 'rb') as f:
                self.model = pickle.load(f)

            vectorizer_file = f'{vectorizer_type}_vectorizer.pkl'
            with open(os.path.join(self.models_dir, vectorizer_file), 'rb') as f:
                self.vectorizer = pickle.load(f)

            return True

        except FileNotFoundError:
            return False

    def load_specific_model(self, model_name, vectorizer_type='tfidf'):
        try:
            model_file = f'{model_name}_{vectorizer_type}.pkl'
            vectorizer_file = f'{vectorizer_type}_vectorizer.pkl'

            with open(os.path.join(self.models_dir, model_file), 'rb') as f:
                self.model = pickle.load(f)

            with open(os.path.join(self.models_dir, vectorizer_file), 'rb') as f:
                self.vectorizer = pickle.load(f)

            return True

        except FileNotFoundError:
            return False

    def predict_single_text(self, text):
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model and vectorizer must be loaded first!")

        processed_text = preprocess_text(text)
        text_vectorized = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(text_vectorized)[0]

        label = "Real" if prediction == 1 else "Fake"

        return {
            'prediction': int(prediction),
            'label': label,
            'processed_text': processed_text
        }

    def predict_batch(self, texts):
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model and vectorizer must be loaded first!")

        results = []
        for text in texts:
            result = self.predict_single_text(text)
            results.append(result)

        return results

    def predict_from_file(self, file_path, text_column='text'):
        df = pd.read_csv(file_path)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the file!")

        texts = df[text_column].astype(str).tolist()
        predictions = self.predict_batch(texts)

        df['predicted_label'] = [pred['label'] for pred in predictions]

        return df

    def get_model_info(self):
        if self.model_info:
            return self.model_info
        else:
            return {
                'model_name': 'Unknown',
                'accuracy': 'Unknown',
                'vectorizer_type': 'Unknown'
            }

    def save_predictions(self, predictions, output_path):
        if isinstance(predictions, list):
            df = pd.DataFrame(predictions)
        else:
            df = predictions

        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

# Example usage and testing
if __name__ == '__main__':
    pipeline = FakeNewsPredictionPipeline()

    if pipeline.load_best_model():
        sample_text = "Breaking news: Scientists discover new planet in our solar system!"
        result = pipeline.predict_single_text(sample_text)

        print("Sample Prediction:")
        print(f"Text: {sample_text}")
        print(f"Prediction: {result['label']}")

        model_info = pipeline.get_model_info()
        print(f"\nModel Info: {model_info}")
    else:
        print("Could not load model. Please run model training first.")
