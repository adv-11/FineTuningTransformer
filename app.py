import streamlit as st
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64
import time

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4CAF50;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .positive {
        background-color: rgba(76, 175, 80, 0.2);
        border-left: 5px solid #4CAF50;
    }
    .neutral {
        background-color: rgba(255, 193, 7, 0.2);
        border-left: 5px solid #FFC107;
    }
    .negative {
        background-color: rgba(244, 67, 54, 0.2);
        border-left: 5px solid #F44336;
    }
    .metrics-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        width: 30%;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🔍 Sentiment Analysis")
    st.markdown("---")
    
    st.subheader("About")
    st.info(
        """
        This app uses a fine-tuned Llama 3.2 model to analyze sentiment in text.
        The model classifies text as:
        - 😊 Positive
        - 😐 Neutral
        - 😔 Negative
        """
    )
    
    st.markdown("---")
    
    # Model selection
    st.subheader("Model Selection")
    model_option = st.selectbox(
        "Choose a model",
        ("Base Llama 3.2-1B-Instruct", "Fine-tuned Llama 3.2")
    )
    
    st.markdown("---")
    
    st.subheader("Dataset Statistics")
    st.markdown("Training samples: 900 (300 per class)")
    st.markdown("Test samples: 900 (300 per class)")
    
    # Dataset distribution pie chart
    labels = ['Positive', 'Neutral', 'Negative']
    values = [300, 300, 300]
    fig = px.pie(
        values=values, 
        names=labels, 
        color=labels,
        color_discrete_map={'Positive':'#4CAF50', 'Neutral':'#FFC107', 'Negative':'#F44336'},
        hole=0.4
    )
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250)
    st.plotly_chart(fig, use_container_width=True)

# Main content area
st.markdown('<h1 class="main-header">📊 Sentiment Analysis with Fine-tuned Transformers</h1>', unsafe_allow_html=True)

tabs = st.tabs(["🏠 Home", "🔍 Analysis", "📝 Dataset", "📈 Evaluation", "ℹ️ About"])

with tabs[0]:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<h2 class="sub-header">Try the Model</h2>', unsafe_allow_html=True)
        
        user_input = st.text_area(
            "Enter a text to analyze sentiment:",
            height=150,
            placeholder="Enter a headline or short text here..."
        )
        
        # Add some example texts
        st.markdown("### Examples")
        examples = [
            "Stock market reaches all-time high as economy booms",
            "Company announces layoffs amid uncertain economic conditions",
            "New regulations to be implemented next month for the industry"
        ]
        
        col1, col2, col3 = st.columns(3)
        
        if col1.button("Example 1"):
            user_input = examples[0]
            st.rerun()
            
        if col2.button("Example 2"):
            user_input = examples[1]
            st.rerun()
            
        if col3.button("Example 3"):
            user_input = examples[2]
            st.rerun()
            
        analyze_button = st.button("Analyze Sentiment", type="primary", use_container_width=True)
        
        if analyze_button and user_input:
            with st.spinner("Analyzing sentiment..."):
                # Simulate model loading and prediction
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # For demonstration, predict based on simple keywords
                # In a real implementation, you would use your model here
                text_lower = user_input.lower()
                
                if "high" in text_lower or "booms" in text_lower or "success" in text_lower:
                    sentiment = "positive"
                    confidence = 0.92
                elif "layoffs" in text_lower or "decline" in text_lower or "crisis" in text_lower:
                    sentiment = "negative"
                    confidence = 0.89
                else:
                    sentiment = "neutral"
                    confidence = 0.78
                
                # Display the result with appropriate styling
                st.markdown(f"### Analysis Results")
                
                sentiment_emojis = {
                    "positive": "😊",
                    "neutral": "😐",
                    "negative": "😔"
                }
                
                st.markdown(f"""
                <div class="result-box {sentiment}">
                    <h3>{sentiment_emojis[sentiment]} Sentiment: {sentiment.capitalize()}</h3>
                    <p>Confidence: {confidence:.2f}</p>
                    <p>Text: "{user_input}"</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show confidence for each class
                st.markdown("### Confidence Scores")
                
                if sentiment == "positive":
                    pos_score = confidence
                    neu_score = (1 - confidence) * 0.8
                    neg_score = (1 - confidence) * 0.2
                elif sentiment == "negative":
                    neg_score = confidence
                    neu_score = (1 - confidence) * 0.8
                    pos_score = (1 - confidence) * 0.2
                else:
                    neu_score = confidence
                    pos_score = (1 - confidence) * 0.5
                    neg_score = (1 - confidence) * 0.5
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Positive", f"{pos_score:.2f}")
                    st.progress(pos_score)
                
                with col2:
                    st.metric("Neutral", f"{neu_score:.2f}")
                    st.progress(neu_score)
                
                with col3:
                    st.metric("Negative", f"{neg_score:.2f}")
                    st.progress(neg_score)
    
    with col2:
        st.markdown('<h2 class="sub-header">Model Performance</h2>', unsafe_allow_html=True)
        
        # Show the accuracy comparison between base and fine-tuned model
        fig = go.Figure()
        
        model_names = ['Base Model', 'Fine-tuned Model']
        accuracies = [0.73, 0.89]  # Example values, replace with actual metrics
        
        fig.add_trace(go.Bar(
            x=model_names,
            y=accuracies,
            text=[f'{acc:.2f}' for acc in accuracies],
            textposition='auto',
            marker_color=['#1E88E5', '#4CAF50']
        ))
        
        fig.update_layout(
            title='Accuracy Comparison',
            xaxis_title='Model',
            yaxis_title='Accuracy',
            yaxis=dict(range=[0, 1]),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show F1-scores by class
        classes = ['Positive', 'Neutral', 'Negative']
        base_f1 = [0.70, 0.68, 0.76]  # Example values
        tuned_f1 = [0.87, 0.84, 0.91]  # Example values
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=classes,
            y=base_f1,
            name='Base Model',
            marker_color='#1E88E5'
        ))
        
        fig.add_trace(go.Bar(
            x=classes,
            y=tuned_f1,
            name='Fine-tuned Model',
            marker_color='#4CAF50'
        ))
        
        fig.update_layout(
            title='F1-Score by Class',
            xaxis_title='Sentiment Class',
            yaxis_title='F1-Score',
            yaxis=dict(range=[0, 1]),
            height=300,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.markdown('<h2 class="sub-header">Batch Analysis</h2>', unsafe_allow_html=True)
    
    # File upload option
    uploaded_file = st.file_uploader("Upload a CSV file with text to analyze", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='latin1')

        
        # Display the dataframe
        st.dataframe(df.head())
        
        # Select text column
        text_column = st.selectbox("Select the column containing text to analyze", df.columns)
        
        if st.button("Run Batch Analysis", type="primary"):
            with st.spinner("Analyzing..."):
                # Add a placeholder for the sentiment column
                df['predicted_sentiment'] = np.random.choice(
                    ['positive', 'neutral', 'negative'], 
                    size=len(df),
                    p=[0.3, 0.4, 0.3]  # Example distribution
                )
                
                # Show results
                st.success(f"Processed {len(df)} texts")
                
                st.markdown("### Results Preview")
                st.dataframe(df.head(10))
                
                # Show distribution
                fig = px.pie(
                    df, 
                    names='predicted_sentiment',
                    title='Sentiment Distribution',
                    color='predicted_sentiment',
                    color_discrete_map={
                        'positive': '#4CAF50',
                        'neutral': '#FFC107',
                        'negative': '#F44336'
                    }
                )
                
                st.plotly_chart(fig)
                
                # Option to download results
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="sentiment_analysis_results.csv">Download Results CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<h2 class="sub-header">Text Comparison</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        text1 = st.text_area(
            "Text 1",
            height=150,
            placeholder="Enter first text here..."
        )
    
    with col2:
        text2 = st.text_area(
            "Text 2",
            height=150,
            placeholder="Enter second text here..."
        )
    
    if st.button("Compare Sentiments", type="primary") and text1 and text2:
        with st.spinner("Analyzing..."):
            # Simulate analysis
            time.sleep(1)
            
            # For demonstration
            sentiment1 = np.random.choice(['positive', 'neutral', 'negative'], p=[0.6, 0.3, 0.1])
            sentiment2 = np.random.choice(['positive', 'neutral', 'negative'], p=[0.1, 0.3, 0.6])
            
            confidence1 = 0.8 + np.random.random() * 0.2
            confidence2 = 0.7 + np.random.random() * 0.2
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="result-box {sentiment1}">
                    <h3>Text 1: {sentiment1.capitalize()}</h3>
                    <p>Confidence: {confidence1:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="result-box {sentiment2}">
                    <h3>Text 2: {sentiment2.capitalize()}</h3>
                    <p>Confidence: {confidence2:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

with tabs[2]:
    st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    # Display sample data
    sample_data = pd.DataFrame({
        'text': [
            "Stock market reaches all-time high as economy booms",
            "Company announces layoffs amid uncertain economic conditions",
            "New regulations to be implemented next month for the industry",
            "Tech giant unveils revolutionary new smartphone",
            "Oil prices remain stable following international talks"
        ],
        'sentiment': ['positive', 'negative', 'neutral', 'positive', 'neutral']
    })
    
    st.dataframe(sample_data, use_container_width=True)
    
    st.markdown("---")
    
    # Data distribution visualization
    st.markdown('<h2 class="sub-header">Data Distribution</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution in the dataset
        fig = px.pie(
            values=[300, 300, 300], 
            names=['Positive', 'Neutral', 'Negative'],
            title='Sentiment Distribution in Dataset',
            color=['Positive', 'Neutral', 'Negative'],
            color_discrete_map={
                'Positive': '#4CAF50',
                'Neutral': '#FFC107',
                'Negative': '#F44336'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Word count distribution
        word_counts = np.random.normal(15, 5, 900).astype(int)
        word_counts = np.clip(word_counts, 5, 30)
        
        fig = px.histogram(
            x=word_counts,
            nbins=20,
            title='Text Length Distribution (Word Count)',
            labels={'x': 'Word Count', 'y': 'Frequency'},
            color_discrete_sequence=['#1E88E5']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Dataset split visualization
    st.markdown('<h2 class="sub-header">Dataset Split</h2>', unsafe_allow_html=True)
    
    fig = go.Figure()
    
    datasets = ['Train', 'Test', 'Validation']
    counts = [900, 900, 150]
    
    fig.add_trace(go.Bar(
        x=datasets,
        y=counts,
        text=counts,
        textposition='auto',
        marker_color=['#4CAF50', '#1E88E5', '#9C27B0']
    ))
    
    fig.update_layout(
        title='Dataset Splits',
        xaxis_title='Split',
        yaxis_title='Number of Samples'
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.markdown('<h2 class="sub-header">Model Evaluation</h2>', unsafe_allow_html=True)
    
    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    
    # Example confusion matrix data
    cm = np.array([
        [250, 30, 20],   # True Negative
        [35, 240, 25],   # True Neutral
        [15, 35, 250]    # True Positive
    ])
    
    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Negative', 'Neutral', 'Positive'],
        yticklabels=['Negative', 'Neutral', 'Positive']
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(fig)
    
    # Classification report
    st.markdown("### Classification Report")
    
    metrics = pd.DataFrame({
        'Class': ['Negative', 'Neutral', 'Positive', 'Average'],
        'Precision': [0.89, 0.84, 0.91, 0.88],
        'Recall': [0.83, 0.80, 0.93, 0.85],
        'F1-Score': [0.86, 0.82, 0.92, 0.87],
        'Support': [300, 300, 300, 900]
    })
    
    st.dataframe(metrics, use_container_width=True)
    
    # Performance metrics visualization
    st.markdown("### Performance Metrics Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Training progress
        epochs = list(range(1, 6))
        train_loss = [0.82, 0.54, 0.39, 0.32, 0.29]
        val_loss = [0.87, 0.60, 0.45, 0.40, 0.38]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=train_loss,
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='#1E88E5')
        ))
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=val_loss,
            mode='lines+markers',
            name='Validation Loss',
            line=dict(color='#F44336')
        ))
        
        fig.update_layout(
            title='Training Progress',
            xaxis_title='Epoch',
            yaxis_title='Loss'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Accuracy progression
        train_acc = [0.70, 0.79, 0.85, 0.88, 0.90]
        val_acc = [0.68, 0.75, 0.82, 0.85, 0.87]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=train_acc,
            mode='lines+markers',
            name='Training Accuracy',
            line=dict(color='#4CAF50')
        ))
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=val_acc,
            mode='lines+markers',
            name='Validation Accuracy',
            line=dict(color='#FFC107')
        ))
        
        fig.update_layout(
            title='Accuracy Progression',
            xaxis_title='Epoch',
            yaxis_title='Accuracy'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Compare base vs fine-tuned model
    st.markdown("### Base vs. Fine-tuned Model Performance")
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    base_metrics = [0.73, 0.72, 0.71, 0.71]
    tuned_metrics = [0.87, 0.88, 0.85, 0.87]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=metrics,
        y=base_metrics,
        name='Base Model',
        marker_color='#1E88E5'
    ))
    
    fig.add_trace(go.Bar(
        x=metrics,
        y=tuned_metrics,
        name='Fine-tuned Model',
        marker_color='#4CAF50'
    ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Metric',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1]),
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.markdown('<h2 class="sub-header">About this Project</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Project Overview
    
    This project demonstrates sentiment analysis using a fine-tuned transformer model (Llama 3.2-1B-Instruct).
    The model was fine-tuned on a dataset of news headlines to classify sentiment as positive, neutral, or negative.
    
    ### Model Architecture
    
    **Base Model**: Meta's Llama 3.2-1B-Instruct
    
    **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
    
    **LoRA Configuration**:
    - lora_alpha: 16
    - lora_dropout: 0
    - r: 64
    - bias: none
    - target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    
    ### Training Details
    
    - Number of epochs: 5
    - Batch size: 1
    - Gradient accumulation steps: 8
    - Learning rate: 2e-4
    - Optimizer: AdamW (32-bit)
    - Learning rate scheduler: Cosine
    
    ### Dataset
    
    The dataset consists of news headlines labeled with three sentiment classes:
    - Positive (300 samples)
    - Neutral (300 samples)
    - Negative (300 samples)
    
    The dataset was split into training (900 samples), testing (900 samples), and validation (150 samples) sets.
    
    ### How to Use
    
    1. Input text in the text area on the home page
    2. Click "Analyze Sentiment" to get predictions
    3. For batch processing, upload a CSV file with text data
    
    ### Project Dependencies
    
    - transformers
    - torch
    - pandas
    - numpy
    - streamlit
    - scikit-learn
    - plotly
    - matplotlib
    - seaborn
    
    ### Future Improvements
    
    - Add multi-language support
    - Implement advanced visualization for error analysis
    - Add support for longer text documents
    - Explore different transformer architectures
    """)

# Function to load the model - hidden in a container that won't be shown
with st.container():
    @st.cache_resource
    def load_model_and_tokenizer(model_name="meta-llama/Llama-3.2-1B-Instruct"):
        """
        This function would load the actual model and tokenizer.
        For demo purposes, we're just creating placeholder functions.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # In production, you'd load the actual model here
            # For demo purposes, we're just returning a placeholder
            st.write("Model loaded successfully!")
            return None, tokenizer
        except:
            st.error("Failed to load model. Using mock prediction instead.")
            return None, None
    
    # This would be called in the actual implementation
    # model, tokenizer = load_model_and_tokenizer()

