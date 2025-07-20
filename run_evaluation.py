from model_training import FakeNewsModelTrainer
from model_evaluation import ModelEvaluator
#from config.path_config import MODEL_DIR, RESULT_DIR


# Step 1: Prepare trainer and train models
trainer = FakeNewsModelTrainer()
trainer.load_and_preprocess_data()
trainer.create_vectorizers()
trainer.split_data()
trainer.train_models()

# Step 2: Evaluate models
evaluator = ModelEvaluator()
results_df, detailed_results = evaluator.run_comprehensive_evaluation(
    trainer.X_count_test,
    trainer.X_tfidf_test,
    trainer.y_test
)

# Step 3: Print results
print("\n Evaluation Summary:")
print(results_df)
