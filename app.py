

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prediction_pipeline import FakeNewsPredictionPipeline
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Fake News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .input-box {
        background-color: #20252C;
        border-radius: 10px;
        padding: 1rem;
        color: white;
        margin-bottom: 20px;
    }
    .result-box {
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 1.1rem;
    }
    .fake-news-result {
        background-color: #ffe6e6;
        border-left: 6px solid #e53935;
        color: #b71c1c;
    }
    .real-news-result {
        background-color: #e6ffe6;
        border-left: 6px solid #43a047;
        color: #1b5e20;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_prediction_pipeline():
    pipeline = FakeNewsPredictionPipeline(models_dir='models')
    if pipeline.load_best_model():
        return pipeline
    else:
        return None

def get_confidence_class(confidence):
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

def create_probability_chart(fake_prob, real_prob):
    fig = go.Figure(data=[
        go.Bar(
            x=['Fake News', 'Real News'],
            y=[fake_prob, real_prob],
            marker_color=['#ff6b6b', '#51cf66'],
            text=[f'{fake_prob:.2%}', f'{real_prob:.2%}'],
            textposition='auto',
        )
    ])
    fig.update_layout(
        title="Prediction Probabilities",
        yaxis_title="Probability",
        xaxis_title="Classification",
        showlegend=False,
        height=400
    )
    return fig

def create_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level (%)"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def main():
    st.markdown('<h1 class="main-header">📰 Fake News Classifier</h1>', unsafe_allow_html=True)
    st.markdown("---")

    pipeline = load_prediction_pipeline()

    if pipeline is None:
        st.error("❌ Could not load the prediction model. Please ensure the model files are available.")
        st.info("💡 Run the model training script first to generate the required model files.")
        return

    model_info = pipeline.get_model_info()
    with st.expander("🔍 Model Information"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Model", model_info.get('model_name', 'Unknown'))
        col2.metric("Accuracy", f"{model_info.get('accuracy', 0):.2%}")
        col3.metric("Vectorizer", model_info.get('vectorizer_type', 'Unknown'))

    st.sidebar.title("🎛️ Navigation")
    mode = st.sidebar.selectbox("Choose prediction mode:", ["Single Text Prediction"])

    if mode == "Single Text Prediction":
        st.markdown("## 📝 Single Text Prediction")
        st.markdown("Use this tool to analyze a news headline or article and detect whether it's likely *Real* or *Fake*.")

        with st.container():
            st.markdown('<div class="input-box">', unsafe_allow_html=True)
            user_input = st.text_area(
                "📥 Enter news text:",
                height=180,
                placeholder="e.g., Drinking bleach can cure COVID-19, says new Harvard study.",
                key="text_input"
            )
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("🔍 Predict", type="primary"):
                if user_input.strip():
                    with st.spinner("Analyzing..."):
                        result = pipeline.predict_single_text(user_input)

                    prediction_class = "fake-news-result" if result['label'] == 'Fake' else "real-news-result"

                    # ✅ Fix for missing 'confidence'
                    confidence = result.get('confidence')
                    confidence_display = f"{confidence:.2%}" if confidence is not None else "N/A"
                    confidence_class = get_confidence_class(confidence) if confidence is not None else ""

                    st.markdown(f'<div class="result-box {prediction_class}">', unsafe_allow_html=True)
                    st.markdown(f"**🎯 Prediction:** {result['label']} News")
                    st.markdown(f"**📊 Confidence:** {confidence_display}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(
                            create_probability_chart(result.get('fake_probability', 0), result.get('real_probability', 0)),
                            use_container_width=True
                        )
                    with col2:
                        if confidence is not None:
                            st.plotly_chart(create_confidence_gauge(confidence), use_container_width=True)

                    with st.expander("🧾 Processed Input Preview"):
                        st.code(result['processed_text'], language="text")
                else:
                    st.warning("⚠️ Please enter some text for analysis.")

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🤖 Powered by Machine Learning | Built with Streamlit</p>
            <p>⚠️ This tool is for educational purposes. Always verify news from multiple reliable sources.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()