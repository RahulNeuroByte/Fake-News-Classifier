
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import os

class ModelEvaluator:
    def __init__(self, models_dir='models', results_dir='results'):  # FILE PATH: Update these paths if folders are in different locations
        """
        Initialize the model evaluator
        
        Args:
            models_dir (str): Directory containing saved models
            results_dir (str): Directory to save evaluation results
        """
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.models = {}
        self.results = {}
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_models(self):
        """Load all saved models"""
        print("Loading models...")
        
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl') and 'vectorizer' not in f and 'info' not in f]
        
        for model_file in model_files:
            model_name = model_file.replace('.pkl', '')
            # FILE PATH: Loading models from models/ directory
            with open(os.path.join(self.models_dir, model_file), 'rb') as f:
                self.models[model_name] = pickle.load(f)
                
        print(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
        
    def load_test_data(self, X_test, y_test):
        """Load test data for evaluation"""
        self.X_test = X_test
        self.y_test = y_test
        
    def evaluate_single_model(self, model_name, model, X_test, y_test):
        """Evaluate a single model and return metrics"""
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
            if y_pred_proba.shape[1] == 2:  # Binary classification
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        else:
            y_pred_proba = None
            auc_score = None
            
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
    def create_confusion_matrix_plot(self, cm, model_name, labels=None):
        """Create and save confusion matrix plot"""
        plt.figure(figsize=(8, 6))
        
        if labels is None:
            labels = ['Fake', 'Real']  # Default labels
            
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # FILE PATH: Confusion matrix plots saved in results/ directory
        plt.savefig(f'{self.results_dir}/confusion_matrix_{model_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_roc_curve_plot(self, y_test, y_pred_proba, model_name):
        """Create and save ROC curve plot"""
        if y_pred_proba is None:
            return
            
        plt.figure(figsize=(8, 6))
        
        if y_pred_proba.shape[1] == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend()
            
            # FILE PATH: ROC curve plots saved in results/ directory
            plt.savefig(f'{self.results_dir}/roc_curve_{model_name}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
    def create_metrics_comparison_plot(self, results_df):
        """Create comparison plot for all metrics"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            if metric in results_df.columns:
                sns.barplot(data=results_df, y='model', x=metric, ax=axes[i], palette='viridis')
                axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
                axes[i].set_xlabel(metric.replace("_", " ").title())
                
                # Add value labels
                for j, v in enumerate(results_df[metric]):
                    if not pd.isna(v):
                        axes[i].text(v + 0.001, j, f'{v:.3f}', va='center')
                        
        plt.tight_layout()
        # FILE PATH: Metrics comparison plot saved in results/ directory
        plt.savefig(f'{self.results_dir}/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_detailed_report(self, results_df):
        """Generate detailed evaluation report"""
        report_content = []
        report_content.append("# Fake News Classification Model Evaluation Report\n")
        report_content.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_content.append("## Model Performance Summary\n")
        
        # Overall summary
        report_content.append("### Top Performing Models\n")
        top_models = results_df.nlargest(3, 'accuracy')
        for idx, row in top_models.iterrows():
            report_content.append(f"**{idx + 1}. {row['model']}**")
            report_content.append(f"   - Accuracy: {row['accuracy']:.4f}")
            report_content.append(f"   - Precision: {row['precision']:.4f}")
            report_content.append(f"   - Recall: {row['recall']:.4f}")
            report_content.append(f"   - F1-Score: {row['f1_score']:.4f}")
            if not pd.isna(row.get('auc_score')):
                report_content.append(f"   - AUC Score: {row['auc_score']:.4f}")
            report_content.append("")
            
        # Detailed metrics table
        report_content.append("### Detailed Metrics Comparison\n")
        report_content.append(results_df.to_markdown(index=False, floatfmt='.4f'))
        report_content.append("\n")
        
        # Model insights
        report_content.append("### Key Insights\n")
        best_model = results_df.loc[results_df['accuracy'].idxmax()]
        report_content.append(f"- **Best Overall Model**: {best_model['model']} with {best_model['accuracy']:.4f} accuracy")
        
        if 'auc_score' in results_df.columns:
            best_auc = results_df.loc[results_df['auc_score'].idxmax()]
            report_content.append(f"- **Best AUC Score**: {best_auc['model']} with {best_auc['auc_score']:.4f} AUC")
            
        # Vectorizer comparison
        count_models = results_df[results_df['model'].str.contains('count')]
        tfidf_models = results_df[results_df['model'].str.contains('tfidf')]
        
        if not count_models.empty and not tfidf_models.empty:
            count_avg = count_models['accuracy'].mean()
            tfidf_avg = tfidf_models['accuracy'].mean()
            better_vectorizer = "TF-IDF" if tfidf_avg > count_avg else "Count Vectorizer"
            report_content.append(f"- **Better Vectorizer**: {better_vectorizer} (Avg Accuracy: {max(count_avg, tfidf_avg):.4f})")
            
        report_content.append("\n### Recommendations\n")
        report_content.append("- Use the best performing model for production deployment")
        report_content.append("- Consider ensemble methods if multiple models perform similarly")
        report_content.append("- Monitor model performance on new data and retrain if necessary")
        
        # Save report
        # FILE PATH: Evaluation report saved in results/ directory
        with open(f'{self.results_dir}/evaluation_report.md', 'w') as f:
            f.write('\n'.join(report_content))
            
        print("Detailed evaluation report generated!")
        
    def run_comprehensive_evaluation(self, X_test_count, X_test_tfidf, y_test):
        """Run comprehensive evaluation on all models"""
        print("Starting comprehensive model evaluation...")
        
        self.load_models()
        
        results_data = []
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            
            # Choose appropriate test data
            if 'count' in model_name:
                X_test = X_test_count
            else:
                X_test = X_test_tfidf
                
            # Evaluate model
            results = self.evaluate_single_model(model_name, model, X_test, y_test)
            
            # Store results
            self.results[model_name] = results
            
            # Add to results data
            results_data.append({
                'model': model_name,
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score'],
                'auc_score': results['auc_score']
            })
            
            # Create visualizations
            self.create_confusion_matrix_plot(results['confusion_matrix'], model_name)
            self.create_roc_curve_plot(y_test, results['probabilities'], model_name)
            
        # Create results DataFrame
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('accuracy', ascending=False)
        
        # Save results
        # FILE PATH: Results saved in results/ directory
        results_df.to_csv(f'{self.results_dir}/detailed_evaluation_results.csv', index=False)
        
        # Create comparison plots
        self.create_metrics_comparison_plot(results_df)
        
        # Generate detailed report
        self.generate_detailed_report(results_df)
        
        print("Comprehensive evaluation completed!")
        print("\nTop 5 Models:")
        print(results_df.head().to_string(index=False))
        
        return results_df, self.results

if __name__ == '__main__':
    # This script requires test data to be provided
    print("Model evaluation script ready. Use this in conjunction with model_training.py")
    print("Example usage:")
    print("evaluator = ModelEvaluator()")
    print("results_df, detailed_results = evaluator.run_comprehensive_evaluation(X_test_count, X_test_tfidf, y_test)")
