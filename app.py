import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# --- 1. CONFIGURATION & DATA LOADING ---
st.set_page_config(page_title="2023 Brand Monitor", layout="wide")

@st.cache_data
def load_data():
    # Load your scraped CSV
    df = pd.read_csv('data.csv')
    # Convert text dates into real Python dates
    df['review_date'] = pd.to_datetime(df['review_date'])
    return df

df = load_data()

# Load the AI Sentiment Model (DistilBERT as requested in homework)
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_analyzer = load_model()

# --- 2. SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Products", "Testimonials", "Reviews"])

# --- 3. PAGE LOGIC ---

if page == "Products":
    st.header("📦 Product Catalog")
    # Show only rows where section is 'product'
    products = df[df['section'] == 'product'][['product_name', 'product_description', 'price']].dropna(subset=['product_name'])
    st.dataframe(products, use_container_width=True)

elif page == "Testimonials":
    st.header("💬 Customer Testimonials")
    # Show only rows where section is 'testimonial'
    testimonials = df[df['section'] == 'testimonial'][['testimonial_text', 'rating']].dropna(subset=['testimonial_text'])
    st.table(testimonials)

elif page == "Reviews":
    st.header("📊 2023 Review Analysis")
    
    # Month Selection Slider (Requirement 2)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    selected_month = st.select_slider("Select Month of 2023", options=months)
    month_index = months.index(selected_month) + 1

    # Filter Data (Requirement 2)
    filtered_reviews = df[
        (df['section'] == 'review') & 
        (df['review_date'].dt.year == 2023) & 
        (df['review_date'].dt.month == month_index)
    ].copy()

    if filtered_reviews.empty:
        st.info(f"No reviews found for {selected_month} 2023.")
    else:
        # Run AI Sentiment Analysis (Requirement 3)
        with st.spinner('AI is analyzing sentiment...'):
            def analyze(text):
                # Clean text and limit length for the model
                res = sentiment_analyzer(str(text)[:512])[0]
                return res['label'], res['score']

            filtered_reviews[['Sentiment', 'Score']] = filtered_reviews['review_text'].apply(lambda x: pd.Series(analyze(x)))

        # Display Metrics (Requirement 4)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", len(filtered_reviews))
        col2.metric("Positive Reviews", len(filtered_reviews[filtered_reviews['Sentiment'] == 'POSITIVE']))
        col3.metric("Avg Confidence", f"{filtered_reviews['Score'].mean():.2%}")

        # Show Table
        st.subheader(f"Reviews for {selected_month}")
        st.dataframe(filtered_reviews[['review_date', 'review_text', 'Sentiment', 'Score']], use_container_width=True)

        # Bar Chart (Requirement 4)
        st.subheader("Sentiment Distribution")
        sentiment_counts = filtered_reviews['Sentiment'].value_counts()
        st.bar_chart(sentiment_counts)

        # WORD CLOUD (Bonus Requirement)
        st.subheader("Review Word Cloud")
        all_text = " ".join(review for review in filtered_reviews.review_text)
        wc = WordCloud(background_color="white", width=800, height=400).generate(all_text)
        
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)