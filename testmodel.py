from prediction_pipeline import FakeNewsPredictionPipeline

pipeline = FakeNewsPredictionPipeline(models_dir='models')

if pipeline.load_best_model():
    print("✅ Model loaded.")
    result = pipeline.predict_single_text("The Prime Minister launched a new digital health program.")
    print("Prediction:", result['label'], "Confidence:", result['confidence'])
else:
    print("❌ Model load failed.")
