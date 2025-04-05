import streamlit as st
import pandas as pd
import requests
import random
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re
import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv(".env")

# Access the API_KEY
API_KEY = os.getenv("API_KEY")

# Initialize session state variables
if 'combined_df' not in st.session_state:
    st.session_state.combined_df = None
if 'youtube_data_fetched' not in st.session_state:
    st.session_state.youtube_data_fetched = False
if 'uploaded_file_data' not in st.session_state:
    st.session_state.uploaded_file_data = None
if 'sentiment_analyzed' not in st.session_state:
    st.session_state.sentiment_analyzed = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# YouTube Data API key

def search_videos(query, max_results=1):
    # Modify the query to include the word 'review'
    modified_query = f"{query} review"
    
    search_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={modified_query}&maxResults={max_results}&relevanceLanguage=en&type=video&key={API_KEY}"
    response = requests.get(search_url)
    
    # Check for errors in response
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        return {}
    
    return response.json()

# Function to get video details (including statistics)
def get_video_details(video_id):
    details_url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={video_id}&key={API_KEY}"
    response = requests.get(details_url)
    return response.json()

# Function to get comments for a specific video ID
def get_video_comments(video_id, max_results=1):
    comments_url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&maxResults={max_results}&key={API_KEY}"
    response = requests.get(comments_url)
    return response.json()

def is_relevant_comment(comment_text):
    """Check if a comment is relevant as a product review or feedback."""
    if not isinstance(comment_text, str):
        return False
        
    text = comment_text.lower()
    
    # Product-related keywords
    review_indicators = [
        'review', 'opinion', 'feedback', 'experience', 'bought', 'purchased', 'using', 'used',
        'quality', 'performance', 'value', 'worth', 'recommend', 'rating', 'stars', 'compare',
        'satisfied', 'dissatisfied', 'issue', 'problem', 'features', 'battery', 'price', 
        'affordable', 'expensive', 'cheap', 'durable', 'reliable', 'lasts', 'broke', 'service', 
        'customer service', 'disappointed', 'happy with', 'works well', "doesn't work", 'pros', 
        'cons', 'awesome', 'terrible', 'good', 'bad', 'better than', 'worse than', 'quality issues'
    ]

    # Irrelevant patterns common in YouTube comments
    irrelevant_patterns = re.compile(
        r'^(please|plz|subscribe|check out|visit|follow|thanks for sharing|click this|watch my|nice video|great video|amazing video|^good job|^well done)',
        re.IGNORECASE
    )
    
    # Filter out short comments and irrelevant patterns
    if len(text.split()) < 5 or irrelevant_patterns.search(text):
        return False

    # Check if comment contains any review indicators
    return any(indicator in text for indicator in review_indicators)

    
def fetch_youtube_data(product_name):
    youtube_data = []
    search_results = search_videos(product_name)
    
    if not search_results.get('items'):
        print("YouTube API returned no results for this product.")
        return pd.DataFrame()
    
    for item in search_results.get('items', []):
        video_id = item['id'].get('videoId')
        if video_id:
            video_details = get_video_details(video_id)
            comments_data = get_video_comments(video_id, max_results=10)
            
            for comment in comments_data.get('items', []):
                comment_text = comment['snippet']['topLevelComment']['snippet']['textDisplay']
                
                # Check comment relevance
                is_relevant = is_relevant_comment(comment_text)
                
                youtube_data.append({
                    'ID': random.randint(1000000, 9999999),
                    'Product_name': product_name,
                    'Review': comment_text,
                    'Source': 'youtube',
                    'Is_Relevant': is_relevant  # Add relevance flag
                })
    
    df = pd.DataFrame(youtube_data)
    
    # Process sentiment based on relevance
    if not df.empty:
        df['Sentiment'] = df.apply(lambda row: 
            get_sentiment(row['Review']) if row['Is_Relevant'] 
            else 'Neutral', axis=1)
        
        df['Sentiment_Score'] = df.apply(lambda row:
            sia.polarity_scores(row['Review'])['compound'] if row['Is_Relevant']
            else 0.0, axis=1)
    
    return df.drop(['Is_Relevant'], axis=1).drop_duplicates(subset=['Review'])


# Download NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Page configuration
st.set_page_config(page_title="Sentiment-Driven Product Development", layout="wide")

# Logo and styling
logo_path = "https://trigent.com/wp-content/uploads/Trigent_Axlr8_Labs.png"
st.markdown(
    f"""
    <div style="text-align:center;">
        <img src="{logo_path}" alt="Trigent Logo" style="max-width:100%;">
    </div>
    """,
    unsafe_allow_html=True
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .dashboard-header {
        text-align: center;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .section-header {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1.5rem 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Dashboard Header
st.markdown("<h1 class='dashboard-header'>Sentiment-Driven Product Development</h1>", unsafe_allow_html=True)
st.write("""
This application is designed to help product development teams, marketing professionals, and data analysts make data-driven decisions by analyzing customer feedback. By examining product reviews, social media mentions, and survey responses, the application classifies feedback sentiment and identifies trends in customer preferences and pain points. This analysis empowers teams to prioritize product improvements, drive customer satisfaction, and align new product features with customer expectations.
""")
# Data preprocessing functions
@st.cache_data
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = pd.Series(text).str.replace(r'https?://\S+|www\.\S+', '', regex=True)[0]
        text = pd.Series(text).str.replace(r'[^\w\s]', '', regex=True)[0]
        text = pd.Series(text).str.replace(r'\d+', '', regex=True)[0]
        return text
    return ""

def get_sentiment(text):
    """Determine sentiment label based on compound score."""
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def analyze_source_metrics(df):
    """Analyze source performance metrics"""
    source_metrics = df.groupby('Source').agg({
        'Sentiment': ['count', 
                     lambda x: (x == 'Positive').mean(),
                     lambda x: (x == 'Negative').mean()],
        'Sentiment_Score': 'mean'
    }).round(3)
    
    source_metrics.columns = ['Total_Reviews', 'Positive_Rate', 'Negative_Rate', 'Avg_Sentiment']
    return source_metrics.sort_values('Positive_Rate', ascending=False)

def normalize_text(text):
    """Normalize text by lowering case and stripping whitespace."""
    return text.lower().strip() if isinstance(text, str) else text

def filter_data(df, product_name=None, sentiments=None, sources=None):
    """Filter data based on product name, sentiment, and source."""
    filtered_df = df.copy()
    
    if product_name and product_name != "All":
        normalized_product_name = str(product_name).lower().strip()
        mask = filtered_df['Product_name'].str.lower().str.contains(
            normalized_product_name.split('(')[0].strip(), 
            case=False, 
            na=False
        )
        filtered_df = filtered_df[mask]

    if sentiments and "All" not in sentiments:
        filtered_df = filtered_df[filtered_df['Sentiment'].isin(sentiments)]
    
    if sources and "All" not in sources:
        filtered_df = filtered_df[filtered_df['Source'].isin(sources)]
    
    return filtered_df


def create_feature_analysis(filtered_df):
    feature_mapping = {
        'quality': [
            # Build Quality
            'build quality', 'manufacturing quality', 'material quality', 'plastic quality',
            'metal build', 'premium build', 'poor build', 'quality',

            # Display Quality
            'display quality', 'screen quality', 'picture quality', 'viewing angles',
            'brightness', 'resolution', 'color accuracy', 'hdr', 'oled', 'lcd', 'ips',

            # Camera Quality (for mobile)
            'camera quality', 'photo quality', 'video quality', 'low light',
            'night mode', 'selfie quality', 'image quality',

            # Sound Quality
            'audio quality', 'sound quality', 'speaker quality', 'mic quality',
            'bass quality', 'microphone quality',

            # General Quality Issues
            'defective', 'faulty', 'damaged', 'broken', 'manufacturing defect',
            'quality control', 'poor quality', 'good quality', 'excellent quality'
        ],
        'price': [
            # Value Perception
            'not value', 'price point', 'price range',
            'budget friendly', 'cost effective', 'affordable price', 'reasonable price',
            'competitive price', 'fair price',

            # Negative Price Terms
            'overpriced', 'expensive', 'costly', 'high price', 'not worth',
            'price too high', 'pricey', 'over budget',

            # Deals/Offers
            'great deal', 'good deal', 'bargain price', 'discount price',
            'sale price', 'offer price'
        ],
        'design': [
            # Physical Attributes
            'form factor', 'design language', 'build design', 'chassis',
            'body design', 'slim design', 'compact design',

            # Aesthetics
            'look', 'appearance', 'aesthetic', 'sleek', 'stylish', 'premium look',
            'modern design', 'elegant', 'finish quality',

            # Size & Weight
            'lightweight', 'heavy', 'bulky', 'portable', 'compact size',
            'screen size', 'dimensions', 'weight distribution',

            # Design Features
            'bezels', 'notch', 'camera bump', 'port placement', 'button placement',
            'fingerprint sensor', 'ergonomic design'
        ],
        'performance': [
            # Processing Power
            'processor speed', 'cpu performance', 'snapdragon', 'mediatek',
            'intel', 'amd', 'gaming performance', 'multitasking',

            # Memory & Storage
            'ram management', 'storage speed', 'memory performance',
            'app loading', 'read speed', 'write speed',

            # Battery Performance
            'battery life', 'battery backup', 'charging speed', 'fast charging',
            'battery drain', 'screen on time', 'standby time',

            # System Performance
            'lag', 'stutter', 'hang', 'freeze', 'smooth performance',
            'responsive', 'quick', 'slow', 'fast', 'optimization',

            # Thermal Performance
            'heating', 'thermal', 'temperature', 'throttling', 'cooling'
        ],
        'durability': [
            # Physical Durability
            'build strength', 'drop resistant', 'scratch resistant',
            'gorilla glass', 'water resistant', 'dust resistant', 'ip rating',

            # Long-term Reliability
            'long lasting', 'reliable', 'sturdy', 'robust', 'solid build',
            'durable', 'wear and tear', 'lifespan',

            # Protection Features
            'protective coating', 'screen protection', 'case friendly',
            'damage resistant', 'shock proof',

            # Durability Issues
            'fragile', 'breaks easily', 'prone to damage', 'weak points',
            'build issues', 'hardware failure'
        ]
    }

    feature_sentiments = []
    
    for main_feature, keywords in feature_mapping.items():
        pattern = '|'.join([main_feature] + keywords)
        mask = filtered_df['Cleaned_Review'].str.contains(pattern, case=False, na=False)
        
        if mask.any():
            feature_data = filtered_df[mask]
            avg_sentiment = feature_data['Sentiment_Score'].mean()
            
            positive_keywords = set()
            negative_keywords = set()
            neutral_keywords = set()
            
            for _, row in feature_data.iterrows():
                review = row['Cleaned_Review'].lower()
                sentiment = row['Sentiment']
                
                found_keywords = [k for k in keywords if k in review]
                
                if found_keywords:
                    if sentiment == 'Positive':
                        positive_keywords.update(found_keywords)
                    elif sentiment == 'Negative':
                        negative_keywords.update(found_keywords)
                    else:
                        neutral_keywords.update(found_keywords)

            # Create more descriptive hover texts
            def format_hover_text(keywords_set, sentiment_type):
                if not keywords_set:
                    return f"No {sentiment_type.lower()} mentions found"
                text = f"<b>{main_feature.title()} - Keywords in {sentiment_type} Reviews:</b><br>"
                return text + "<br>".join([f"• {k.title()}" for k in sorted(keywords_set)])

            positive_hover = format_hover_text(positive_keywords, "Positive")
            negative_hover = format_hover_text(negative_keywords, "Negative")
            neutral_hover = format_hover_text(neutral_keywords, "Neutral")

            feature_sentiments.append({
                'Feature': main_feature.title(),
                'Average Sentiment': avg_sentiment,
                'Positive Count': len(positive_keywords),
                'Negative Count': len(negative_keywords),
                'Neutral Count': len(neutral_keywords),
                'Positive Hover': positive_hover,
                'Negative Hover': negative_hover,
                'Neutral Hover': neutral_hover
            })

    if feature_sentiments:
        feature_df = pd.DataFrame(feature_sentiments)

        fig = go.Figure()

        # Stacked bar chart with sentiment-specific hover text
        fig.add_trace(go.Bar(
            x=feature_df['Feature'],
            y=feature_df['Positive Count'],
            name='Positive Mentions',
            marker_color='#2ecc71',
            yaxis='y2',
            opacity=0.8,
            hoverinfo='text',
            hovertext=feature_df['Positive Hover'],
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        ))

        fig.add_trace(go.Bar(
            x=feature_df['Feature'],
            y=feature_df['Negative Count'],
            name='Negative Mentions',
            marker_color='#e74c3c',
            yaxis='y2',
            opacity=0.8,
            hoverinfo='text',
            hovertext=feature_df['Negative Hover'],
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        ))

        fig.add_trace(go.Bar(
            x=feature_df['Feature'],
            y=feature_df['Neutral Count'],
            name='Neutral Mentions',
            marker_color='#95a5a6',
            yaxis='y2',
            opacity=0.8,
            hoverinfo='text',
            hovertext=feature_df['Neutral Hover'],
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        ))

        # Line chart for average sentiment
        fig.add_trace(go.Scatter(
            x=feature_df['Feature'],
            y=feature_df['Average Sentiment'],
            name='Average Sentiment',
            marker_color='#FF9800',
            mode='lines+markers',
            line=dict(width=2),
            yaxis='y',
            hoverinfo='text',
            hovertext=[f"Average Sentiment: {val:.2f}" for val in feature_df['Average Sentiment']],
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        ))

        fig.update_layout(
            title='Feature Analysis: Sentiment and Keyword Mentions',
            yaxis=dict(
                title='Average Sentiment',
                titlefont=dict(color='#FF9800'),
                side='left'
            ),
            yaxis2=dict(
                title='Mention Count',
                titlefont=dict(color='#333333'),
                side='right',
                overlaying='y'
            ),
            barmode='stack',
            showlegend=True,
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        return fig
    return None

def generate_insights(reviews_text):
    headers = {
        'Content-Type': 'application/json',
        'api-key': '51ba5d46601c477b844d3883af93463c'
    }
    data = {
        "messages": [
            {"role": "system", "content": "Analyze the following product reviews and provide recommendations and insights for improvement, focusing on specific issues and potential enhancements:\n\n{reviews_text}\n\nInsights:"},
            {"role": "user", "content": reviews_text}  # Using reviews_text here directly
        ],
        "max_tokens": 800,
        "temperature": 0.5
    }

    response = requests.post(
        'https://genai-trigent-openai.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview',
        headers=headers, json=data)
    
    if response.status_code == 200:
        content = response.json()["choices"][0]["message"]["content"]
        return content
    else:
        return f"Error: {response.status_code} - {response.text}"


def chunk_reviews(reviews, chunk_size=100):
    """Yield successive chunks from reviews."""
    for i in range(0, len(reviews), chunk_size):
        yield reviews[i:i + chunk_size]

def process_uploaded_file(uploaded_file):
    if uploaded_file is not None and (st.session_state.uploaded_file_data is None):
        st.session_state.uploaded_file_data = pd.read_csv(uploaded_file)
    return st.session_state.uploaded_file_data

def get_enhanced_sentiment(text):
    """
    Enhanced sentiment analysis that:
    1. Breaks text into sentences for granular analysis
    2. Weighs negative statements more heavily
    3. Considers certain keywords as strong sentiment indicators
    4. Accounts for common patterns in product reviews
    """
    if not isinstance(text, str):
        return "Neutral", 0.0
    
    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Define strong negative indicators commonly found in product reviews
    strong_negative_patterns = [
        "don't buy",
        "major issue",
        "doesn't work",
        "not working",
        "disappointed",
        "bug",
        "glitch",
        "problem",
        "broken",
        "terrible",
        "waste",
        "poor",
        "defective",
        "failure",
        "not helpful",
        "unusable"
    ]
    
    # Define phrases that often indicate a review is describing problems
    problem_indicators = [
        "customer support",
        "service center",
        "facing this issue",
        "many people",
        "needs fixing",
        "doesn't help",
        "didn't solve",
        "regret",
        "hot",
        "lag",
        "problem",
        "dissatisfied"
    ]
    
    # Initial sentiment scoring
    compound_scores = []
    
    # Split into sentences for granular analysis
    sentences = text.split('.')
    
    # Track presence of strong negative indicators
    has_strong_negative = False
    has_problem_indicator = False
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        # Check for strong negative patterns
        for pattern in strong_negative_patterns:
            if pattern.lower() in sentence.lower():
                has_strong_negative = True
                break
                
        # Check for problem indicators
        for indicator in problem_indicators:
            if indicator.lower() in sentence.lower():
                has_problem_indicator = True
                break
        
        # Get VADER sentiment for the sentence
        scores = sia.polarity_scores(sentence)
        compound_scores.append(scores['compound'])
    
    # Calculate base sentiment
    if compound_scores:
        avg_compound = sum(compound_scores) / len(compound_scores)
    else:
        return "Neutral", 0.0
    
    # Adjust sentiment based on detected patterns
    if has_strong_negative:
        avg_compound -= 0.3  # Substantial negative adjustment
    
    if has_problem_indicator:
        avg_compound -= 0.2  # Moderate negative adjustment
    
    # Count negative vs positive sentences
    negative_sentences = sum(1 for score in compound_scores if score < -0.05)
    positive_sentences = sum(1 for score in compound_scores if score > 0.05)
    
    # If there are more negative sentences than positive ones, ensure overall sentiment is negative
    if negative_sentences > positive_sentences and avg_compound > -0.05:
        avg_compound -= 0.2
    
    # Determine final sentiment label
    if avg_compound >= 0.05:
        sentiment = "Positive"
    elif avg_compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return sentiment, round(avg_compound, 4)

# Main app logic
st.markdown("<div class='section-header'>1. Dataset Preview Data Cleaning</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Flipkart CSV", type="csv")
flipkart_df = process_uploaded_file(uploaded_file)

if flipkart_df is not None:

    st.write("Dataset preview")
    st.dataframe(flipkart_df[['ID', 'Product_name', 'Review','Source']])
    product_name = st.text_input("Search for a product (e.g., Samsung Galaxy A33):")
    
    # YouTube data fetch button
    if st.button("Fetch YouTube Comments") and product_name:
        with st.spinner("Fetching YouTube comments..."):
            youtube_df = fetch_youtube_data(product_name)
            
            # Ensure youtube_df has the expected structure
            if 'ID' not in youtube_df.columns:
                youtube_df['ID'] = ''  # Create an empty 'ID' column if missing
            if youtube_df.empty:
                st.write("No relevant YouTube comments found for the specified product.")
            else:
                youtube_df['ID'] = youtube_df['ID'].astype(str)
                st.session_state.combined_df = pd.concat([flipkart_df, youtube_df], ignore_index=True)
                st.session_state.youtube_data_fetched = True

                # Process the combined data immediately
                st.session_state.combined_df['Cleaned_Review'] = st.session_state.combined_df['Review'].apply(preprocess_text)
                # st.session_state.combined_df['Sentiment'] = st.session_state.combined_df['Cleaned_Review'].apply(get_sentiment)
                # In your main app code, replace:
                st.session_state.combined_df['Sentiment'], st.session_state.combined_df['Sentiment_Score'] = zip(*st.session_state.combined_df['Cleaned_Review'].apply(get_enhanced_sentiment))
                st.session_state.combined_df['Sentiment_Score'] = st.session_state.combined_df['Cleaned_Review'].apply(
                    lambda x: sia.polarity_scores(x)['compound']
                )
                st.session_state.processed_data = st.session_state.combined_df.copy()
                st.session_state.sentiment_analyzed = True


    # Display data and filters only if we have processed data
    if st.session_state.processed_data is not None:
        # Display the combined dataset
        st.write("Combined Data Preview:")
        st.dataframe(st.session_state.processed_data[['ID', 'Product_name', 'Review', 'Source']])

        # Section 2: Data Cleaning
        st.markdown("<div class='section-header'>2. Data Cleaning</div>", unsafe_allow_html=True)
        st.write(st.session_state.processed_data[['ID','Product_name', 'Review', 'Cleaned_Review']])

        # Section 3: Sentiment Analysis
        st.markdown("<div class='section-header'>3. Sentiment Analysis</div>", unsafe_allow_html=True)
        sentiment_analysis_df = st.session_state.processed_data[['ID','Product_name', 'Review', 'Sentiment', 'Sentiment_Score']]
        st.write(sentiment_analysis_df)

        # Search and Filter Options
        st.markdown("<div class='section-header'>4. Search and Filter Options</div>", unsafe_allow_html=True)
        
        # Create filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            product_filter_options = st.session_state.processed_data['Product_name'].unique()
            product_filter = st.selectbox(
                "Filter by Product Name",
                options=["All"] + list(product_filter_options),
                key='product_filter'
            )
        
        with col2:
            sentiment_filter_options = ["Positive", "Negative", "Neutral"]
            sentiment_filter = st.multiselect(
                "Filter by Sentiment",
                options=["All"] + sentiment_filter_options,
                default=["All"],
                key='sentiment_filter'
            )
        
        with col3:
            source_filter_options = st.session_state.processed_data['Source'].unique()
            source_filter = st.multiselect(
                "Filter by Source",
                options=["All"] + list(source_filter_options),
                default=["All"],
                key='source_filter'
            )
        
        # Apply filters
        filtered_data = filter_data(
            st.session_state.processed_data,
            product_name=product_filter,
            sentiments=sentiment_filter,
            sources=source_filter
        )
        
        # Display filtered data
        if not filtered_data.empty:
            st.write(filtered_data[['ID', 'Product_name', 'Review', 'Source', 'Sentiment', 'Sentiment_Score']])
            
            top_reviews = filtered_data.nlargest(20, 'Sentiment_Score')['Review'].tolist()
            reviews_text = "\n".join(top_reviews)
            
            if st.button("Generate Insights"):
                with st.spinner("Generating insights..."):
                    insights = generate_insights(reviews_text)
                    if insights:
                        st.write("### Recommendations and Insights")
                        st.write(insights)

            # Sentiment Distribution
            st.markdown("<div class='section-header'>5. Sentiment Distribution</div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment_counts = filtered_data['Sentiment'].value_counts()
                fig_bar = px.bar(
                    sentiment_counts,
                    x=sentiment_counts.index,
                    y=sentiment_counts.values,
                    labels={'x': 'Sentiment', 'y': 'Number of Reviews'},
                    title='Distribution of Sentiments'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                fig_pie = px.pie(
                    sentiment_counts,
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title='Proportion of Sentiments',
                    color=sentiment_counts.index,
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Source Analysis
            st.markdown("<div class='section-header'>Source Analysis</div>", unsafe_allow_html=True)
            source_metrics = analyze_source_metrics(filtered_data)
            
            if not source_metrics.empty:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    source_fig = go.Figure()
                    sources = source_metrics.index
                    
                    source_fig.add_trace(go.Bar(
                        name='Positive Rate',
                        x=sources,
                        y=source_metrics['Positive_Rate'] * 100,
                        marker_color='#2ecc71'
                    ))
                    
                    source_fig.add_trace(go.Bar(
                        name='Negative Rate',
                        x=sources,
                        y=source_metrics['Negative_Rate'] * 100,
                        marker_color='#e74c3c'
                    ))
                    
                    source_fig.update_layout(
                        title='Source Performance Comparison',
                        yaxis=dict(title='Percentage', tickformat='.1f'),
                        barmode='group'
                    )
                    
                    st.plotly_chart(source_fig, use_container_width=True)
                
                with col2:
                    # Display best performing sources with their positive rates
                    st.markdown("#### Best Performing Sources")
                    for idx, (source, metrics) in enumerate(source_metrics.iterrows(), 1):
                        positive_rate = metrics['Positive_Rate'] * 100
                        st.write(f"{idx}. {source}: {positive_rate:.1f}% positive")

                    # Highlight the overall best source
                    best_source = source_metrics.index[0]
                    best_rate = source_metrics.iloc[0]['Positive_Rate'] * 100
                    st.success(f"Top Source by Positive Rate: {best_source} ({best_rate:.1f}%)")

            
            # Feature Analysis
            st.markdown("<div class='section-header'>6. Feature Analysis</div>", unsafe_allow_html=True)
            feature_fig = create_feature_analysis(filtered_data)
            if feature_fig:
                st.plotly_chart(feature_fig, use_container_width=True)
            else:
                st.write("No feature mentions found in the filtered data.")

            # Download Button
            st.download_button(
                label="📥 Download Results",
                data=filtered_data.to_csv(index=False).encode('utf-8'),
                file_name=f'sentiment_analysis_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
            )
        else:
            st.write("No data available after applying filters.")


footer_html = """
   <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
   <div style="text-align: center;">
       <p>
           Copyright © 2024 | <a href="https://trigent.com/ai/" target="_blank" aria-label="Trigent Website">Trigent Software Inc.</a> All rights reserved. |
           <a href="https://www.linkedin.com/company/trigent-software/" target="_blank" aria-label="Trigent LinkedIn"><i class="fab fa-linkedin"></i></a> |
           <a href="https://www.twitter.com/trigentsoftware/" target="_blank" aria-label="Trigent Twitter"><i class="fab fa-twitter"></i></a> |
           <a href="https://www.youtube.com/channel/UCNhAbLhnkeVvV6MBFUZ8hOw" target="_blank" aria-label="Trigent Youtube"><i class="fab fa-youtube"></i></a>
       </p>
   </div>
   """

footer_css = """
   <style>
   .footer {
       position: fixed;
       z-index: 1000;
       left: 0;
       bottom: 0;
       width: 100%;
       background-color: white;
       color: black;
       text-align: center;
   }
   [data-testid="stSidebarNavItems"] {
       max-height: 100%!important;
   }
   [data-testid="collapsedControl"] {
       display: none;
   }
   </style>
   """

footer = f"{footer_css}<div class='footer'>{footer_html}</div>"

st.markdown(footer, unsafe_allow_html=True)