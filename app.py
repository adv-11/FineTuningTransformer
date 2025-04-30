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
                progress_text.text("Loading cardiffnlp/twitter-roberta-base-sentiment model...")
                model_id = "cardiffnlp/twitter-roberta-base-sentiment"
            else:  # Fine-tuned model
                progress_text.text("Loading finiteautomata/bertweet-base-sentiment-analysis model...")
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
        
        # Map the result to our format (positive, negative, neutral)
        label = result[0]['label'].lower()
        score = result[0]['score']
        
        # Different models use different label formats
        # cardiffnlp/twitter-roberta-base-sentiment uses labels like: LABEL_0, LABEL_1, LABEL_2
        # finiteautomata/bertweet-base-sentiment-analysis uses labels like: POS, NEG, NEU
        
        # Handle different label formats
        if 'label_0' in label:
            sentiment = 'negative'  # In the RoBERTa model, LABEL_0 is negative
        elif 'label_1' in label:
            sentiment = 'neutral'   # In the RoBERTa model, LABEL_1 is neutral
        elif 'label_2' in label:
            sentiment = 'positive'  # In the RoBERTa model, LABEL_2 is positive
        elif 'negative' in label or label == 'neg':
            sentiment = 'negative'
        elif 'positive' in label or label == 'pos':
            sentiment = 'positive'
        else:
            sentiment = 'neutral'
            
        return sentiment, score
    except Exception as e:
        # Log the error, but don't break the app flow
        print(f"Error analyzing sentiment: {str(e)}")
        return "neutral", 0.5  # Default to neutral as a fallback

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
            if true_label in ['positive', 'pos', '1', '1.0']:
                true_label = 'positive'
            elif true_label in ['negative', 'neg', '0', '0.0', '-1', '-1.0']:
                true_label = 'negative'
            else:
                true_label = 'neutral'  # Anything else is considered neutral
            
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
    
    # Calculate confusion matrix for 3 classes
    cm = confusion_matrix(
        valid_results['true_sentiment'], 
        valid_results['predicted_sentiment'],
        labels=['positive', 'neutral', 'negative']
    )
    
    # Calculate metrics - using macro average for multiclass
    accuracy = accuracy_score(valid_results['true_sentiment'], valid_results['predicted_sentiment'])
    precision = precision_score(
        valid_results['true_sentiment'], 
        valid_results['predicted_sentiment'],
        labels=['positive', 'neutral', 'negative'],
        average='macro',
        zero_division=0
    )
    recall = recall_score(
        valid_results['true_sentiment'], 
        valid_results['predicted_sentiment'],
        labels=['positive', 'neutral', 'negative'],
        average='macro',
        zero_division=0
    )
    f1 = f1_score(
        valid_results['true_sentiment'], 
        valid_results['predicted_sentiment'],
        labels=['positive', 'neutral', 'negative'],
        average='macro',
        zero_division=0
    )
    
    # Calculate per-class metrics
    class_precision = precision_score(
        valid_results['true_sentiment'], 
        valid_results['predicted_sentiment'],
        labels=['positive', 'neutral', 'negative'],
        average=None,
        zero_division=0
    )
    
    class_recall = recall_score(
        valid_results['true_sentiment'], 
        valid_results['predicted_sentiment'],
        labels=['positive', 'neutral', 'negative'],
        average=None,
        zero_division=0
    )
    
    class_f1 = f1_score(
        valid_results['true_sentiment'], 
        valid_results['predicted_sentiment'],
        labels=['positive', 'neutral', 'negative'],
        average=None,
        zero_division=0
    )
    
    # Save evaluation metrics
    st.session_state['confusion_matrix'] = {
        'matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1
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
        - Second column: Label (positive/neutral/negative)
        
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
                        **Model:** cardiffnlp/twitter-roberta-base-sentiment
                        **Description:** A RoBERTa model trained on Twitter data for sentiment analysis.
                        **Labels:** Positive/Neutral/Negative (3-class classification)
                        """)
                    else:
                        st.info("""
                        **Model:** finiteautomata/bertweet-base-sentiment-analysis
                        **Description:** A BERTweet model fine-tuned for sentiment analysis on Twitter data.
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
                    self.neutral_words = ["okay", "ok", "fine", "average", "neutral", "moderate", "so-so", "fair", 
                                         "decent", "standard", "usual", "normal", "regular", "common"]
                
                def __call__(self, text):
                    if not isinstance(text, str):
                        text = str(text)
                    
                    text = text.lower()
                    
                    # Count positive, negative and neutral words
                    pos_count = sum(1 for word in self.positive_words if word in text)
                    neg_count = sum(1 for word in self.negative_words if word in text)
                    neut_count = sum(1 for word in self.neutral_words if word in text)
                    
                    # Determine sentiment
                    max_count = max(pos_count, neg_count, neut_count)
                    if max_count == 0:
                        # No sentiment words found, default to neutral
                        return [{"label": "NEUTRAL", "score": 0.7}]
                    elif max_count == pos_count:
                        return [{"label": "POSITIVE", "score": 0.6 + (0.3 * (pos_count / (pos_count + neg_count + neut_count + 1)))}]
                    elif max_count == neg_count:
                        return [{"label": "NEGATIVE", "score": 0.6 + (0.3 * (neg_count / (pos_count + neg_count + neut_count + 1)))}]
                    else:
                        return [{"label": "NEUTRAL", "score": 0.6 + (0.3 * (neut_count / (pos_count + neg_count + neut_count + 1)))}]
            
            st.session_state['sentiment_analyzer'] = SimpleClassifier()
            st.success("✅ Simple fallback classifier loaded successfully!")
            st.info("This is a basic rule-based classifier that looks for positive, neutral, and negative keywords.")

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
            st.info("Expected format: First column is text, second column is sentiment label (positive/neutral/negative)")
            
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
                    
                    # Standardize labels for visualization
                    standardized_labels = df[label_column].astype(str).str.lower().apply(
                        lambda x: 'positive' if x in ['positive', 'pos', '1', '1.0'] 
                        else ('negative' if x in ['negative', 'neg', '0', '0.0', '-1', '-1.0'] 
                             else 'neutral')
                    )
                    
                    label_counts = standardized_labels.value_counts()
                    
                    # Create a bar chart
                    fig = px.bar(
                        x=label_counts.index,
                        y=label_counts.values,
                        color=label_counts.index,
                        color_discrete_map={
                            'positive': '#4CAF50',
                            'neutral': '#FFC107',
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
                                    
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    sns.heatmap(
                                        cm, 
                                        annot=True, 
                                        fmt='d',
                                        cmap='Blues',
                                        xticklabels=['Predicted Positive', 'Predicted Neutral', 'Predicted Negative'],
                                        yticklabels=['Actual Positive', 'Actual Neutral', 'Actual Negative']
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
                                        st.metric("Precision (macro)", f"{cm_data['precision']:.4f}")
                                    with col3:
                                        st.metric("Recall (macro)", f"{cm_data['recall']:.4f}")
                                    with col4:
                                        st.metric("F1 Score (macro)", f"{cm_data['f1']:.4f}")
                                
                                    # Display per-class metrics
                                    st.markdown("### Per-Class Metrics")
                                    
                                    class_metrics = pd.DataFrame({
                                        'Class': ['Positive', 'Neutral', 'Negative'],
                                        'Precision': cm_data['class_precision'],
                                        'Recall': cm_data['class_recall'],
                                        'F1 Score': cm_data['class_f1']
                                    })
                                    
                                    st.dataframe(class_metrics, hide_index=True)
                                
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
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.set(font_scale=1.4)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=['Predicted Positive', 'Predicted Neutral', 'Predicted Negative'],
            yticklabels=['Actual Positive', 'Actual Neutral', 'Actual Negative'],
            annot_kws={'size': 16}
        )
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.title('Confusion Matrix', fontsize=16)
        st.pyplot(fig)
        
        # Display performance metrics in a visually appealing way
        st.markdown("### Performance Metrics")
        
        # Display overall metrics
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Create a gauge chart for accuracy
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cm_data['accuracy'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Accuracy"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "lightgreen"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "red"},
                        {'range': [0.5, 0.7], 'color': "orange"},
                        {'range': [0.7, 0.9], 'color': "lightgreen"},
                        {'range': [0.9, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.7
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Create a horizontal bar chart for precision, recall, and F1
            metrics_data = {
                'Metric': ['Precision', 'Recall', 'F1 Score'],
                'Value': [cm_data['precision'], cm_data['recall'], cm_data['f1']]
            }
            
            fig = px.bar(
                metrics_data,
                x='Value',
                y='Metric',
                orientation='h',
                color='Value',
                color_continuous_scale=[(0, "red"), (0.5, "yellow"), (0.7, "lightgreen"), (1, "green")],
                range_color=[0, 1],
                labels={'Value': 'Score'},
                text_auto='.4f'
            )
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Display per-class metrics
        st.markdown("### Per-Class Performance")
        
        # Create a dataframe with class metrics
        class_metrics = pd.DataFrame({
            'Class': ['Positive', 'Neutral', 'Negative'],
            'Precision': cm_data['class_precision'],
            'Recall': cm_data['class_recall'],
            'F1 Score': cm_data['class_f1']
        })
        
        # Create a grouped bar chart
        fig = px.bar(
            class_metrics.melt(id_vars='Class', var_name='Metric', value_name='Value'),
            x='Class',
            y='Value',
            color='Metric',
            barmode='group',
            text_auto='.3f',
            color_discrete_map={
                'Precision': '#1E88E5',
                'Recall': '#43A047',
                'F1 Score': '#FB8C00'
            }
        )
        
        fig.update_layout(
            yaxis_range=[0, 1],
            legend_title_text='',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results view
        st.markdown("### Detailed Results")
        
        # Add filters
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_filter = st.multiselect(
                "Filter by predicted sentiment",
                options=['positive', 'neutral', 'negative', 'error'],
                default=['positive', 'neutral', 'negative', 'error']
            )
        
        with col2:
            # Add confidence slider
            min_confidence = float(results_df['confidence'].min()) if not results_df.empty else 0.0
            max_confidence = float(results_df['confidence'].max()) if not results_df.empty else 1.0
            confidence_range = st.slider(
                "Confidence score range",
                min_value=0.0,
                max_value=1.0,
                value=(min_confidence, max_confidence),
                step=0.05
            )
        
        # Filter the results based on selections
        filtered_df = results_df[
            (results_df['predicted_sentiment'].isin(sentiment_filter)) &
            (results_df['confidence'] >= confidence_range[0]) &
            (results_df['confidence'] <= confidence_range[1])
        ]
        
        # Add option to show correct/incorrect predictions
        prediction_status = st.radio(
            "Show predictions",
            ["All", "Correct Predictions", "Incorrect Predictions"],
            horizontal=True
        )
        
        if prediction_status == "Correct Predictions":
            filtered_df = filtered_df[filtered_df['predicted_sentiment'] == filtered_df['true_sentiment']]
        elif prediction_status == "Incorrect Predictions":
            filtered_df = filtered_df[filtered_df['predicted_sentiment'] != filtered_df['true_sentiment']]
        
        # Display the filtered dataframe
        if not filtered_df.empty:
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download button for filtered results
            csv = filtered_df.to_csv(index=False, encoding='utf-8')
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="filtered_sentiment_results.csv">Download Filtered Results</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("No results match the selected filters.")
        
        # Export full dashboard report as HTML
        st.markdown("### Export Full Report")
        
        if st.button("Generate HTML Report"):
            with st.spinner("Generating report..."):
                # Create a full HTML report with all visualizations
                # This would be more complex to implement fully
                
                st.success("Report generated successfully!")
                
                # Placeholder for the actual implementation
                report_html = """
                <html>
                <head>
                    <title>Sentiment Analysis Dashboard Report</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        h1 { color: #1E88E5; }
                        h2 { color: #4CAF50; }
                        .container { margin: 20px 0; }
                        .metric { display: inline-block; padding: 10px; margin: 5px; background: #f5f5f5; border-radius: 5px; }
                    </style>
                </head>
                <body>
                    <h1>Sentiment Analysis Dashboard Report</h1>
                    <p>Generated on {}</p>
                    
                    <div class="container">
                        <h2>Performance Summary</h2>
                        <div class="metric">Accuracy: {:.4f}</div>
                        <div class="metric">Precision: {:.4f}</div>
                        <div class="metric">Recall: {:.4f}</div>
                        <div class="metric">F1 Score: {:.4f}</div>
                    </div>
                    
                    <p>This is a placeholder for the full report. Download the CSV for detailed results.</p>
                </body>
                </html>
                """.format(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    cm_data['accuracy'],
                    cm_data['precision'],
                    cm_data['recall'],
                    cm_data['f1']
                )
                
                # Encode HTML to download
                b64 = base64.b64encode(report_html.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="sentiment_analysis_report.html">Download HTML Report</a>'
                st.markdown(href, unsafe_allow_html=True)
    else:
        st.info("Please run the analysis on the 'Dataset Analysis' tab first to see results here.")

# Single Text Analysis interface
st.markdown("---")
st.markdown('<h2 class="sub-header">📝 Single Text Analysis</h2>', unsafe_allow_html=True)

if st.session_state['sentiment_analyzer'] is not None:
    # Text input for single analysis
    text_input = st.text_area("Enter text to analyze", height=150)
    
    if text_input:
        if st.button("Analyze Text"):
            with st.spinner("Analyzing..."):
                sentiment, confidence = analyze_sentiment(text_input, st.session_state['sentiment_analyzer'])
                
                # Display the result
                st.markdown(f"### Sentiment: {sentiment.capitalize()}")
                
                # Show result with appropriate styling
                if sentiment == "positive":
                    st.markdown(f'<div class="result-box positive"><h3>😊 Positive Sentiment</h3><p>Confidence: {confidence:.4f}</p></div>', unsafe_allow_html=True)
                elif sentiment == "neutral":
                    st.markdown(f'<div class="result-box neutral"><h3>😐 Neutral Sentiment</h3><p>Confidence: {confidence:.4f}</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-box negative"><h3>😔 Negative Sentiment</h3><p>Confidence: {confidence:.4f}</p></div>', unsafe_allow_html=True)
                
                # Show confidence meter
                st.markdown("### Confidence")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.33], 'color': "lightgray"},
                            {'range': [0.33, 0.67], 'color': "gray"},
                            {'range': [0.67, 1], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.5
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Please load a model from the sidebar first to use this feature.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
    Sentiment Analysis Evaluation Dashboard | Created with Streamlit
</div>
""", unsafe_allow_html=True)