import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
import base64
import time
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis Evaluation Dashboard",
    page_icon="📊",
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

# Initialize session state for storing models and metrics
if 'sentiment_analyzer' not in st.session_state:
    st.session_state['sentiment_analyzer'] = None
if 'evaluation_metrics' not in st.session_state:
    st.session_state['evaluation_metrics'] = None
if 'batch_results' not in st.session_state:
    st.session_state['batch_results'] = None
if 'confusion_matrix' not in st.session_state:
    st.session_state['confusion_matrix'] = None

# Function to load the sentiment analysis model
@st.cache_resource
def load_sentiment_model(model_name):
    """Load the sentiment analysis model from Hugging Face."""
    try:
        with st.spinner(f"Loading {model_name}..."):
            # Show a more detailed loading message
            progress_text = st.empty()
            
            # Set model based on selection
            if model_name == "Base Model":
                progress_text.text("Loading DistilBERT model...")
                model_id = "distilbert-base-uncased-finetuned-sst-2-english"
            else:  # Fine-tuned model
                progress_text.text("Loading BERTweet model...")
                model_id = "finiteautomata/bertweet-base-sentiment-analysis"
            
            # Add error handling for offline or connection issues
            try:
                # Add explicit device placement for better performance
                device = -1  # CPU by default
                
                # Try to use GPU if available
                progress_text.text(f"Loading {model_id}...")
                sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model=model_id,
                    device=device
                )
                
                # Test the model with a simple example
                progress_text.text("Testing model with sample text...")
                test_result = sentiment_analyzer("This is a test.")
                
                # Clear the progress text
                progress_text.empty()
                
                return sentiment_analyzer
                
            except Exception as conn_error:
                progress_text.empty()
                raise Exception(f"Connection error: {str(conn_error)}")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.markdown("""
        ### Troubleshooting:
        - Check your internet connection
        - Try a different model
        - If using GPU, ensure it has enough memory
        """)
        return None

# Function to analyze sentiment
def analyze_sentiment(text, model):
    """Analyze the sentiment of the given text using the loaded model."""
    try:
        # Handle empty text or non-string input
        if not isinstance(text, str) or text.strip() == '':
            return "neutral", 0.5
        
        # Clean the text of any encoding issues before processing
        # Replace any problematic characters that might cause issues
        clean_text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Limit text length if it's too long (some models have max token limits)
        max_length = 512  # Most models handle this length well
        if len(clean_text) > max_length:
            clean_text = clean_text[:max_length]
        
        # Process the text with the model
        result = model(clean_text)
        
        # Map the result to our format (positive, negative)
        # Different models may have different label formats
        label = result[0]['label'].lower()
        score = result[0]['score']
        
        # Map the label to our standard format
        if 'positive' in label or label == 'pos':
            sentiment = 'positive'
        else:
            sentiment = 'negative'
            
        return sentiment, score
    except Exception as e:
        # Log the error, but don't break the app flow
        print(f"Error analyzing sentiment: {str(e)}")
        return "negative", 0.5  # Default to negative as a fallback

# Function to process batch sentiment analysis and evaluate against true labels
def batch_analyze_sentiment(df, text_column, label_column, model):
    """Run sentiment analysis on a batch of texts in a dataframe and compare with true labels."""
    results = []
    predicted_sentiments = []
    scores = []
    true_sentiments = []
    processed_texts = []
    errors = []
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process each text in the dataframe
    total_rows = len(df)
    
    # Track errors for reporting
    error_count = 0
    success_count = 0
    
    # Create a container for intermediate results
    result_container = st.empty()
    
    # For large datasets, use batch processing
    batch_size = 10  # Process in batches for responsiveness
    
    for i in range(0, total_rows, batch_size):
        batch_end = min(i + batch_size, total_rows)
        
        # Process each text in the current batch
        for j in range(i, batch_end):
            # Get the text and true label
            text = df.iloc[j][text_column]
            true_label = df.iloc[j][label_column].lower()  # Ensure lowercase for consistency
            
            # Standardize the true label format
            if true_label == 'pos' or true_label == '1' or true_label == 1:
                true_label = 'positive'
            elif true_label == 'neg' or true_label == '0' or true_label == 0:
                true_label = 'negative'
            
            # Update the progress bar
            progress = int((j + 1) / total_rows * 100)
            progress_bar.progress(progress)
            status_text.text(f"Processing {j+1}/{total_rows} ({progress}%)")
            
            # Analyze sentiment
            try:
                predicted_sentiment, score = analyze_sentiment(text, model)
                predicted_sentiments.append(predicted_sentiment)
                scores.append(score)
                true_sentiments.append(true_label)
                processed_texts.append(text[:100] + "..." if len(str(text)) > 100 else str(text))
                errors.append(None)
                success_count += 1
            except Exception as e:
                predicted_sentiments.append("error")
                scores.append(0.0)
                true_sentiments.append(true_label)
                processed_texts.append(str(text)[:50] + "..." if len(str(text)) > 50 else str(text))
                errors.append(str(e))
                error_count += 1
        
        # Show intermediate results (update every batch)
        if (i + batch_size) % (batch_size * 5) == 0 or batch_end == total_rows:
            with result_container.container():
                st.write(f"Processed: {batch_end}/{total_rows} texts")
                st.write(f"Success: {success_count}, Errors: {error_count}")
    
    # Clear the progress indicators
    progress_bar.empty()
    status_text.empty()
    result_container.empty()
    
    # Add results to the dataframe
    result_df = df.copy()
    result_df['predicted_sentiment'] = predicted_sentiments
    result_df['confidence'] = scores
    result_df['true_sentiment'] = true_sentiments
    
    # Calculate confusion matrix
    # Filter out errors
    valid_results = result_df[result_df['predicted_sentiment'] != 'error']
    
    # Calculate confusion matrix
    cm = confusion_matrix(
        valid_results['true_sentiment'], 
        valid_results['predicted_sentiment'],
        labels=['positive', 'negative']
    )
    
    # Calculate metrics
    accuracy = accuracy_score(valid_results['true_sentiment'], valid_results['predicted_sentiment'])
    precision = precision_score(
        valid_results['true_sentiment'], 
        valid_results['predicted_sentiment'],
        pos_label='positive'
    )
    recall = recall_score(
        valid_results['true_sentiment'], 
        valid_results['predicted_sentiment'],
        pos_label='positive'
    )
    f1 = f1_score(
        valid_results['true_sentiment'], 
        valid_results['predicted_sentiment'],
        pos_label='positive'
    )
    
    # Save evaluation metrics
    st.session_state['confusion_matrix'] = {
        'matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    # Create a detailed error log for download if needed
    if error_count > 0:
        error_log = pd.DataFrame({
            'text': processed_texts,
            'predicted_sentiment': predicted_sentiments,
            'true_sentiment': true_sentiments,
            'confidence': scores,
            'error': errors
        })
        
        # Filter to show only errors
        error_log = error_log[error_log['error'].notnull()]
        
        if len(error_log) > 0:
            st.warning(f"⚠️ There were {error_count} errors during processing. See details below.")
            with st.expander("Error Details"):
                st.dataframe(error_log)
                
                # Option to download error log
                csv = error_log.to_csv(index=False, encoding='utf-8')
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="sentiment_analysis_errors.csv">Download Error Log</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    return result_df

# Sidebar
with st.sidebar:
    st.title("🔍 Sentiment Analysis")
    st.markdown("---")
    
    st.subheader("About")
    st.info(
        """
        This app uses Hugging Face transformer models to analyze sentiment in text.
        Upload your dataset with two columns:
        - First column: Text
        - Second column: Label (positive/negative)
        
        The app will calculate accuracy metrics and confusion matrix.
        """
    )
    
    st.markdown("---")
    
    # Model selection
    st.subheader("Model Selection")
    model_option = st.selectbox(
        "Choose a model",
        ("Base Model", "Fine-tuned Model")
    )
    
    # Load the selected model
    if st.button("Load Selected Model"):
        # Show loading indicators
        with st.spinner("Loading model..."):
            # Try to load the model
            try:
                st.session_state['sentiment_analyzer'] = load_sentiment_model(model_option)
                if st.session_state['sentiment_analyzer']:
                    st.success(f"✅ {model_option} loaded successfully!")
                    # Add model info
                    if model_option == "Base Model":
                        st.info("""
                        **Model:** DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
                        **Description:** A smaller, faster version of BERT fine-tuned on the Stanford Sentiment Treebank.
                        **Labels:** Positive/Negative (binary classification)
                        """)
                    else:
                        st.info("""
                        **Model:** BERTweet (finiteautomata/bertweet-base-sentiment-analysis)
                        **Description:** A RoBERTa model trained on Twitter data and fine-tuned for sentiment analysis.
                        **Labels:** Positive/Neutral/Negative (3-class classification)
                        """)
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
                st.info("Please try again or select a different model.")
    
    # Show warning if no model is loaded
    if st.session_state['sentiment_analyzer'] is None:
        st.warning("⚠️ Please load a model to continue.")
        
    # Add model fallback option
    with st.expander("Can't load models?"):
        st.markdown("""
        If you're having trouble loading the Hugging Face models, try using the fallback option below.
        This will use a simple rule-based classifier that doesn't require downloading large models.
        """)
        
        if st.button("Use Simple Fallback Classifier"):
            # Create a simple rule-based classifier as fallback
            class SimpleClassifier:
                def __init__(self):
                    self.positive_words = ["good", "great", "excellent", "positive", "happy", "joy", "love", "wonderful", 
                                          "amazing", "fantastic", "delighted", "success", "successful", "boom", "growth"]
                    self.negative_words = ["bad", "awful", "terrible", "negative", "sad", "angry", "hate", "poor", "terrible",
                                          "horrible", "disappointing", "failure", "crash", "crisis", "decline", "layoff"]
                
                def __call__(self, text):
                    if not isinstance(text, str):
                        text = str(text)
                    
                    text = text.lower()
                    
                    # Count positive and negative words
                    pos_count = sum(1 for word in self.positive_words if word in text)
                    neg_count = sum(1 for word in self.negative_words if word in text)
                    
                    # Determine sentiment
                    if pos_count > neg_count:
                        return [{"label": "POSITIVE", "score": 0.7 + (0.2 * (pos_count / (pos_count + neg_count + 1)))}]
                    else:
                        return [{"label": "NEGATIVE", "score": 0.7 + (0.2 * (neg_count / (pos_count + neg_count + 1)))}]
            
            st.session_state['sentiment_analyzer'] = SimpleClassifier()
            st.success("✅ Simple fallback classifier loaded successfully!")
            st.info("This is a basic rule-based classifier that looks for positive and negative keywords.")

# Main content area
st.markdown('<h1 class="main-header">📊 Sentiment Analysis Evaluation</h1>', unsafe_allow_html=True)

tabs = st.tabs(["📁 Dataset Analysis", "📊 Results"])

with tabs[0]:
    st.markdown('<h2 class="sub-header">Dataset Analysis</h2>', unsafe_allow_html=True)
    
    # File upload option
    uploaded_file = st.file_uploader("Upload a CSV file with text and labels", type=["csv"])
    
    if uploaded_file is not None:
        # Encoding options
        st.markdown("### File Encoding")
        st.info("If you encounter encoding errors, try selecting a different encoding.")
        encoding_options = ["utf-8", "latin1", "ISO-8859-1", "cp1252", "utf-16", "utf-32"]
        selected_encoding = st.selectbox("Select file encoding", encoding_options)
        
        # Additional error handling options
        error_handling = st.radio(
            "Error handling for problematic characters",
            ["strict", "ignore", "replace"],
            index=2,
            help="'strict': raise error, 'ignore': skip problematic characters, 'replace': replace with replacement character"
        )
        
        # Try to load the file with the selected encoding
        try:
            df = pd.read_csv(uploaded_file, encoding=selected_encoding, encoding_errors=error_handling)
            
            # Display info about required format
            st.info("Expected format: First column is text, second column is sentiment label (positive/negative)")
            
            # Display the dataframe
            st.markdown("### Data Preview")
            st.dataframe(df.head())
            
            # Display some stats about the data
            st.markdown("### Data Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isna().sum().sum())
            
            # Get column names
            columns = df.columns.tolist()
            
            # Check if there are at least 2 columns
            if len(columns) < 2:
                st.error("Your dataset must have at least 2 columns: text and label")
            else:
                # Auto-select the first and second columns
                text_column = columns[0]
                label_column = columns[1]
                
                st.markdown("### Column Selection")
                col1, col2 = st.columns(2)
                with col1:
                    text_column = st.selectbox("Text column", columns, index=columns.index(text_column))
                with col2:
                    label_column = st.selectbox("Label column", columns, index=columns.index(label_column))
                
                # Check for missing values in the selected columns
                missing_text = df[text_column].isna().sum()
                missing_label = df[label_column].isna().sum()
                
                if missing_text > 0 or missing_label > 0:
                    st.warning(
                        f"⚠️ Missing values detected: {missing_text} in text column, {missing_label} in label column. "
                        "These rows will be skipped during analysis."
                    )
                
                # Pre-process text option
                clean_text = st.checkbox("Clean text before analysis (remove special characters, extra spaces)", value=True)
                
                # Check label distribution
                try:
                    # Get unique labels
                    unique_labels = df[label_column].astype(str).str.lower().unique()
                    st.markdown("### Label Distribution")
                    
                    # Standardize labels
                    label_counts = df[label_column].astype(str).str.lower().map(
                        lambda x: 'positive' if x in ['positive', 'pos', '1', '1.0'] else 'negative'
                    ).value_counts()
                    
                    # Create a bar chart
                    fig = px.bar(
                        x=label_counts.index,
                        y=label_counts.values,
                        color=label_counts.index,
                        color_discrete_map={
                            'positive': '#4CAF50',
                            'negative': '#F44336'
                        },
                        labels={'x': 'Sentiment', 'y': 'Count'}
                    )
                    st.plotly_chart(fig)
                    
                    # Show text length distribution
                    st.markdown("### Text Length Distribution")
                    
                    # Calculate text lengths
                    text_lengths = df[text_column].astype(str).apply(len)
                    
                    # Create a histogram
                    fig = px.histogram(
                        x=text_lengths,
                        nbins=30,
                        labels={'x': 'Text Length (characters)', 'y': 'Count'},
                        color_discrete_sequence=['#1E88E5']
                    )
                    st.plotly_chart(fig)
                    
                except Exception as e:
                    st.error(f"Error analyzing labels: {str(e)}")
                
                # Run analysis button
                if st.button("Run Analysis", type="primary"):
                    if st.session_state['sentiment_analyzer'] is None:
                        st.error("Please load a model first from the sidebar.")
                    else:
                        # Clean text if selected
                        if clean_text:
                            with st.spinner("Cleaning text..."):
                                # Make a copy to avoid modifying the original
                                analysis_df = df.copy()
                                # Basic text cleaning - replace special characters with space and multiple spaces with single space
                                analysis_df[text_column] = analysis_df[text_column].astype(str).str.replace(r'[^\w\s]', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
                        else:
                            analysis_df = df.copy()
                        
                        # Drop rows with missing values
                        analysis_df = analysis_df.dropna(subset=[text_column, label_column])
                        
                        # Ensure text column is string type
                        analysis_df[text_column] = analysis_df[text_column].astype(str)
                        
                        # Process the batch
                        with st.spinner("Analyzing sentiment..."):
                            try:
                                # Process the batch
                                results_df = batch_analyze_sentiment(analysis_df, text_column, label_column, st.session_state['sentiment_analyzer'])
                                
                                # Save results to session state
                                st.session_state['batch_results'] = results_df
                                
                                # Show results
                                st.success(f"✅ Successfully processed {len(results_df)} texts")
                                
                                st.markdown("### Results Preview")
                                st.dataframe(results_df.head(10))
                                
                                # Show confusion matrix
                                if st.session_state['confusion_matrix'] is not None:
                                    cm_data = st.session_state['confusion_matrix']
                                    
                                    st.markdown("### Confusion Matrix")
                                    
                                    # Format confusion matrix
                                    cm = cm_data['matrix']
                                    
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    sns.heatmap(
                                        cm, 
                                        annot=True, 
                                        fmt='d',
                                        cmap='Blues',
                                        xticklabels=['Predicted Positive', 'Predicted Negative'],
                                        yticklabels=['Actual Positive', 'Actual Negative']
                                    )
                                    plt.ylabel('True Label')
                                    plt.xlabel('Predicted Label')
                                    plt.title('Confusion Matrix')
                                    st.pyplot(fig)
                                    
                                    # Display metrics
                                    st.markdown("### Performance Metrics")
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric("Accuracy", f"{cm_data['accuracy']:.4f}")
                                    with col2:
                                        st.metric("Precision", f"{cm_data['precision']:.4f}")
                                    with col3:
                                        st.metric("Recall", f"{cm_data['recall']:.4f}")
                                    with col4:
                                        st.metric("F1 Score", f"{cm_data['f1']:.4f}")
                                
                                # Option to download results
                                csv = results_df.to_csv(index=False, encoding='utf-8')
                                b64 = base64.b64encode(csv.encode()).decode()
                                href = f'<a href="data:file/csv;base64,{b64}" download="sentiment_analysis_results.csv">Download Results CSV</a>'
                                st.markdown(href, unsafe_allow_html=True)
                            
                            except Exception as e:
                                st.error(f"Error during analysis: {str(e)}")
                                st.info("Try cleaning the text, selecting different columns, or changing the encoding.")
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.markdown("""
            ### Troubleshooting Tips:
            1. Try a different encoding (common ones: utf-8, latin1, cp1252)
            2. Check if your CSV file has proper formatting
            3. Try opening the file in a text editor and saving it with UTF-8 encoding
            4. If it's a large file, try with a smaller sample first
            """)
            
            # Advanced debugging info (collapsible)
            with st.expander("Advanced Error Details"):
                st.code(str(e))
                st.markdown("**Common encoding errors:**")
                st.markdown("- `'utf-8' codec can't decode byte`: File is not in UTF-8 format. Try latin1 or cp1252.")
                st.markdown("- `'charmap' codec can't decode byte`: Character mapping issue. Try latin1.")
                st.markdown("- `'ascii' codec can't decode byte`: ASCII encoding can't handle special characters. Try utf-8 or latin1.")

with tabs[1]:
    st.markdown('<h2 class="sub-header">Results and Metrics</h2>', unsafe_allow_html=True)
    
    if st.session_state['batch_results'] is not None and st.session_state['confusion_matrix'] is not None:
        results_df = st.session_state['batch_results']
        cm_data = st.session_state['confusion_matrix']
        
        # Display confusion matrix
        st.markdown("### Confusion Matrix")
        
        # Create an annotated heatmap of the confusion matrix
        cm = cm_data['matrix']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.set(font_scale=1.4)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=['Predicted Positive', 'Predicted Negative'],
            yticklabels=['Actual Positive', 'Actual Negative'],
            annot_kws={'size': 16}
        )
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.title('Confusion Matrix', fontsize=16)
        st.pyplot(fig)
        
        # Display performance metrics in a visually appealing way
        st.markdown("### Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create a gauge chart for accuracy
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cm_data['accuracy'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Accuracy"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "#1E88E5"},
                    'steps': [
                        {'range': [0, 0.6], 'color': "#EF5350"},
                        {'range': [0.6, 0.8], 'color': "#FFCA28"},
                        {'range': [0.8, 1], 'color': "#66BB6A"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig)
        
        with col2:
            # Create a radar chart for precision, recall, and F1
            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=[cm_data['precision'], cm_data['recall'], cm_data['f1']],
                theta=['Precision', 'Recall', 'F1 Score'],
                fill='toself',
                name='Metrics',
                fillcolor='rgba(76, 175, 80, 0.2)',
                line_color='#4CAF50'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig)
        
        # Display detailed metrics
        st.markdown("### Detailed Metrics")
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [
                cm_data['accuracy'], 
                cm_data['precision'], 
                cm_data['recall'], 
                cm_data['f1']
            ]
        })
        
        st.dataframe(metrics_df, hide_index=True)
        
        # Error analysis section
        st.markdown("### Error Analysis")
        
        # Find examples of false positives and false negatives
        false_positives = results_df[
            (results_df['predicted_sentiment'] == 'positive') & 
            (results_df['true_sentiment'] == 'negative')
        ].sort_values('confidence', ascending=False)
        
        false_negatives = results_df[
            (results_df['predicted_sentiment'] == 'negative') & 
            (results_df['true_sentiment'] == 'positive')
        ].sort_values('confidence', ascending=False)
        
        # Display false positives and false negatives
        error_tabs = st.tabs(["False Positives", "False Negatives"])
        
        with error_tabs[0]:
            st.write(f"Total False Positives: {len(false_positives)}")
            if len(false_positives) > 0:
                st.markdown("Top examples where model incorrectly predicted POSITIVE:")
                st.dataframe(false_positives.head(10)[[text_column, 'confidence']])
            else:
                st.info("No false positives found.")
        
        with error_tabs[1]:
            st.write(f"Total False Negatives: {len(false_negatives)}")
            if len(false_negatives) > 0:
                st.markdown("Top examples where model incorrectly predicted NEGATIVE:")
                st.dataframe(false_negatives.head(10)[[text_column, 'confidence']])
            else:
                st.info("No false negatives found.")

# Add visualization for confidence distribution
st.markdown("### Confidence Distribution")

fig = px.histogram(
    results_df,
    x="confidence",
    color="predicted_sentiment",
    marginal="box",
    nbins=30,
    color_discrete_map={
        "positive": "#4CAF50",
        "negative": "#F44336"
    },
    labels={"confidence": "Confidence Score", "predicted_sentiment": "Predicted Sentiment"},
    title="Confidence Score Distribution by Predicted Sentiment"
)
st.plotly_chart(fig)

# Add export options
st.markdown("### Export Results")

# Generate downloadable CSV
csv = results_df.to_csv(index=False, encoding='utf-8')
b64 = base64.b64encode(csv.encode()).decode()
href = f'<a href="data:file/csv;base64,{b64}" download="sentiment_analysis_results.csv">Download Complete Results</a>'
st.markdown(href, unsafe_allow_html=True)

# Generate a detailed report
if st.button("Generate Detailed Report"):
    with st.spinner("Creating detailed report..."):
        # Create a new dataframe for the report
        report_df = pd.DataFrame({
            "Metric": ["Total Samples", "Correct Predictions", "Incorrect Predictions", 
                    "Accuracy", "Precision", "Recall", "F1 Score",
                    "True Positives", "True Negatives", 
                    "False Positives", "False Negatives"],
            "Value": [
                len(results_df),
                len(results_df[results_df["predicted_sentiment"] == results_df["true_sentiment"]]),
                len(results_df[results_df["predicted_sentiment"] != results_df["true_sentiment"]]),
                cm_data["accuracy"], 
                cm_data["precision"], 
                cm_data["recall"], 
                cm_data["f1"],
                cm[0][0],  # True positives
                cm[1][1],  # True negatives
                cm[1][0],  # False positives
                cm[0][1]   # False negatives
            ]
        })
        
        st.dataframe(report_df, hide_index=True)
        
        # Create downloadable report
        report_csv = report_df.to_csv(index=False, encoding='utf-8')
        b64_report = base64.b64encode(report_csv.encode()).decode()
        href_report = f'<a href="data:file/csv;base64,{b64_report}" download="sentiment_analysis_report.csv">Download Metrics Report</a>'
        st.markdown(href_report, unsafe_allow_html=True)
        
        st.success("Report generated successfully!")
else:
    # Show an interactive table of results
    st.markdown("### Interactive Results Explorer")
    
    # Add filters
    st.markdown("Filter by:")
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_filter = st.multiselect(
            "Predicted Sentiment",
            options=["positive", "negative"],
            default=["positive", "negative"]
        )
    
    with col2:
        agreement_filter = st.radio(
            "Prediction Agreement",
            options=["All", "Correct Predictions", "Incorrect Predictions"]
        )
    
    # Apply filters
    filtered_df = results_df.copy()
    
    if sentiment_filter:
        filtered_df = filtered_df[filtered_df["predicted_sentiment"].isin(sentiment_filter)]
    
    if agreement_filter == "Correct Predictions":
        filtered_df = filtered_df[filtered_df["predicted_sentiment"] == filtered_df["true_sentiment"]]
    elif agreement_filter == "Incorrect Predictions":
        filtered_df = filtered_df[filtered_df["predicted_sentiment"] != filtered_df["true_sentiment"]]
    
    # Show the filtered results
    st.dataframe(filtered_df[[text_column, "predicted_sentiment", "true_sentiment", "confidence"]])
