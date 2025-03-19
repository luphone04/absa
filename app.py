import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import requests
import json
from collections import Counter

# ------------------------------------------------------
# Hugging Face API Settings (Replace with your own token)
# ------------------------------------------------------
api_key = "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"  # Replace with your actual token
headers = {"Authorization": f"Bearer {api_key}"}

# Endpoints for Laptop domain
laptop_aspect_extraction_url = "https://hnhp22gmf9aa0lsn.us-east-1.aws.endpoints.huggingface.cloud"
laptop_sentiment_classification_url = "https://hgr9a67mny2lebgw.us-east-1.aws.endpoints.huggingface.cloud"

# Endpoints for Restaurant domain
restaurant_aspect_extraction_url = "https://pjb2wusut7gazhlw.us-east-1.aws.endpoints.huggingface.cloud"
restaurant_sentiment_classification_url = "https://ea62drh96w58z855.us-east-1.aws.endpoints.huggingface.cloud"

# ----------------------------------
# Mock Inference Functions (fallback)
# ----------------------------------
def mock_aspect_extraction(review_text):
    """Simulate aspect extraction by returning fixed aspect: sentiment pairs."""
    return "battery life: positive, screen: negative, design: positive"

def mock_aspect_sentiment_classification(review_text, aspect):
    """Simulate aspect-sentiment classification by returning a random sentiment."""
    sentiments = ["positive", "negative", "neutral"]
    return random.choice(sentiments)

# ----------------------------------
# Hugging Face API Functions (Custom Endpoints)
# ----------------------------------
def call_aspect_extraction_custom(review_text, extraction_url):
    payload = {"inputs": review_text}
    response = requests.post(extraction_url, json=payload, headers=headers)
    if response.status_code != 200:
        st.error(f"Aspect Extraction endpoint error: {response.status_code}")
        return ""
    result = response.json()
    return result[0].get("predicted", "")

def call_sentiment_classification_custom(review_text, aspect, sentiment_url):
    payload = {"inputs": f"{review_text} The aspect is {aspect}."}
    response = requests.post(sentiment_url, json=payload, headers=headers)
    if response.status_code != 200:
        st.error(f"Sentiment Classification endpoint error: {response.status_code}")
        return ""
    result = response.json()
    return result[0].get("predicted", "")

# ----------------------------------
# Data Loading and Dashboard Functions
# ----------------------------------
def load_reviews(profile):
    """Return a DataFrame for a given profile."""
    if profile.lower() == "laptop":
        data = {
            "review_text": [
                "The battery life is excellent, but the keyboard is unresponsive.",
                "I love the sleek design and vibrant display, but the performance is subpar."
            ],
            "extracted_aspects": [
                "battery life: positive, keyboard: negative",
                "design: positive, display: positive, performance: negative"
            ]
        }
        return pd.DataFrame(data)
    elif profile.lower() == "restaurant":
        data = {
            "review_text": [
                "The service was outstanding and the food was delicious.",
                "The ambiance is cozy, but the wait time was too long."
            ],
            "extracted_aspects": [
                "service: positive, food: positive",
                "ambiance: positive, wait time: negative"
            ]
        }
        return pd.DataFrame(data)
    elif profile.lower() == "tripadvisor":
        try:
            df = pd.read_csv("tripadvisor_reviews_sentiment.csv")
            if "review_text" not in df.columns:
                if "raw_text" in df.columns:
                    df.rename(columns={"raw_text": "review_text"}, inplace=True)
                elif "Review" in df.columns:
                    df.rename(columns={"Review": "review_text"}, inplace=True)
                else:
                    st.error("Column 'review_text' not found. Available columns: " + ", ".join(df.columns))
            if "extracted_aspects" not in df.columns:
                if "aspectTerm" in df.columns and "predicted_sentiment" in df.columns:
                    df["extracted_aspects"] = df.apply(
                        lambda row: f"{row['aspectTerm']}: {row['predicted_sentiment']}", axis=1
                    )
                else:
                    st.error("Column 'extracted_aspects' not found and cannot be created. Available columns: " + ", ".join(df.columns))
            return df
        except Exception as e:
            st.error(f"Error loading TripAdvisor data: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

def parse_extracted_aspects(extracted_aspects_str):
    pairs = []
    if pd.isnull(extracted_aspects_str):
        return pairs
    for pair in extracted_aspects_str.split(","):
        if ":" in pair:
            aspect, sentiment = pair.split(":", 1)
            pairs.append((aspect.strip(), sentiment.strip()))
    return pairs

def compute_dashboard_metrics(df):
    sentiments_list = []
    aspect_list = []
    for aspects_str in df["extracted_aspects"]:
        pairs = parse_extracted_aspects(aspects_str)
        for aspect, sentiment in pairs:
            sentiments_list.append(sentiment.lower())
            aspect_list.append(aspect.lower())
    sentiment_counts = Counter(sentiments_list)
    total_sentiments = sum(sentiment_counts.values())
    sentiment_percentages = {s: (count / total_sentiments) * 100 for s, count in sentiment_counts.items()} if total_sentiments > 0 else {}
    aspect_counts = Counter(aspect_list)
    total_aspects = sum(aspect_counts.values())
    top_aspects = [(aspect, count / total_aspects) for aspect, count in aspect_counts.most_common(10)]
    return sentiment_counts, sentiment_percentages, top_aspects

def color_sentiment(sentiment):
    s_lower = sentiment.lower()
    if s_lower == "positive":
        return f"<span style='color:green;font-weight:bold'>{sentiment.capitalize()}</span>"
    elif s_lower == "negative":
        return f"<span style='color:red;font-weight:bold'>{sentiment.capitalize()}</span>"
    else:
        return f"<span style='color:orange;font-weight:bold'>{sentiment.capitalize()}</span>"

def plot_sentiment_donut(sentiment_counts, sentiment_percentages):
    sentiments = list(sentiment_counts.keys())
    counts = list(sentiment_counts.values())
    def make_label(sentiment):
        return f"{sentiment.capitalize()} ({sentiment_percentages[sentiment]:.1f}%)"
    label_texts = [make_label(s) for s in sentiments]
    color_map = {"positive": "#4CAF50", "negative": "#F44336", "neutral": "#FF9800"}
    colors = [color_map.get(s, "#9E9E9E") for s in sentiments]
    explode = [0.02] * len(counts)
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, _, _ = ax.pie(
        counts, labels=None, autopct='%1.1f%%', startangle=90,
        colors=colors, explode=explode, pctdistance=0.85,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    ax.axis('equal')
    ax.legend(wedges, label_texts, title="Sentiments", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    st.pyplot(fig)

def plot_aspect_bar(top_aspects):
    if not top_aspects:
        st.write("No aspects found.")
        return
    aspects, frequencies = zip(*top_aspects)
    fig, ax = plt.subplots()
    ax.bar(aspects, frequencies, color="#1976D2")
    ax.set_xlabel("Aspects")
    ax.set_ylabel("Frequency (Decimal)")
    ax.set_title("Top Mentioned Aspects")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def get_all_aspects_reviews(df):
    aspect_reviews_map = {}
    for _, row in df.iterrows():
        pairs = parse_extracted_aspects(row["extracted_aspects"])
        for aspect, sentiment in pairs:
            aspect_lower = aspect.lower()
            if aspect_lower not in aspect_reviews_map:
                aspect_reviews_map[aspect_lower] = []
            aspect_reviews_map[aspect_lower].append((row["review_text"], sentiment))
    return aspect_reviews_map

# ----------------------------------
# Main App Layout with Multi-Page Dashboard
# ----------------------------------
st.set_page_config(page_title="ABSA Demo", layout="wide")
st.title("Aspect-Based Sentiment Analysis Demo")

# Main task selection: Dashboard vs Live Inference.
main_task = st.sidebar.selectbox("Select Task", ["Dashboard", "Live Inference"])

if main_task == "Dashboard":
    st.header("Dashboard")
    st.write("Select a profile to view aspect-sentiment insights from reviews.")
    
    profile = st.selectbox("Select Profile", ["Laptop", "Restaurant", "TripAdvisor"])
    df_profile = load_reviews(profile)
    
    if df_profile.empty:
        st.error("No data available to display.")
    else:
        if profile.lower() in ["laptop", "restaurant"]:
            if st.button("Run Inference on Reviews"):
                st.write("Running aspect extraction inference...")
                df_profile["extracted_aspects"] = df_profile["review_text"].apply(mock_aspect_extraction)
                st.success("Inference complete!")
        else:
            st.info("Using pre-computed aspect extraction results for TripAdvisor.")
        
        if "extracted_aspects" in df_profile.columns:
            sentiment_counts, sentiment_percentages, top_aspects = compute_dashboard_metrics(df_profile)
            dashboard_page = st.sidebar.selectbox("Select Dashboard Page", 
                                                  ["Sentiment Distribution", "Top Aspects", "Reviews by Aspect"])
            if dashboard_page == "Sentiment Distribution":
                st.write("## Overall Sentiment Distribution")
                plot_sentiment_donut(sentiment_counts, sentiment_percentages)
            elif dashboard_page == "Top Aspects":
                st.write("## Most Frequent Mentioned Aspects")
                plot_aspect_bar(top_aspects)
            elif dashboard_page == "Reviews by Aspect":
                st.write("## View Reviews by Aspect")
                aspect_reviews_map = get_all_aspects_reviews(df_profile)
                if aspect_reviews_map:
                    all_aspects = sorted(aspect_reviews_map.keys())
                    selected_aspect = st.selectbox("Choose an aspect", options=all_aspects)
                    polarity_options = ["positive", "negative", "neutral"]
                    selected_polarities = st.multiselect("Filter by sentiment polarity", 
                                                         options=polarity_options,
                                                         default=polarity_options)
                    keyword_filter = st.text_input("Filter by keyword (optional)")
                    if selected_aspect:
                        reviews_for_aspect = aspect_reviews_map[selected_aspect]
                        if selected_polarities:
                            reviews_for_aspect = [
                                (review, sentiment) for review, sentiment in reviews_for_aspect
                                if sentiment.lower() in selected_polarities
                            ]
                        if keyword_filter:
                            keyword_filter_lower = keyword_filter.lower()
                            reviews_for_aspect = [
                                (review, sentiment) for review, sentiment in reviews_for_aspect
                                if keyword_filter_lower in review.lower()
                            ]
                        if reviews_for_aspect:
                            sentiments = [s.lower() for _, s in reviews_for_aspect]
                            sentiment_counts_aspect = Counter(sentiments)
                            total_reviews = sum(sentiment_counts_aspect.values())
                            pos_percent = (sentiment_counts_aspect.get("positive", 0) / total_reviews) * 100 if total_reviews > 0 else 0
                            neg_percent = (sentiment_counts_aspect.get("negative", 0) / total_reviews) * 100 if total_reviews > 0 else 0
                            neu_percent = (sentiment_counts_aspect.get("neutral", 0) / total_reviews) * 100 if total_reviews > 0 else 0
                            
                            st.markdown(f"**Sentiment Distribution for '{selected_aspect}' Reviews:**")
                            st.markdown(
                                f"Positive: {pos_percent:.1f}% | Negative: {neg_percent:.1f}% | Neutral: {neu_percent:.1f}%",
                            )
                            
                            st.write(f"**Reviews mentioning '{selected_aspect}' with selected filters:**")
                            for review_text, sentiment in reviews_for_aspect:
                                # Minimal HTML is used for coloring.
                                if sentiment.lower() == "positive":
                                    sentiment_html = "<span style='color:green;font-weight:bold'>Positive</span>"
                                elif sentiment.lower() == "negative":
                                    sentiment_html = "<span style='color:red;font-weight:bold'>Negative</span>"
                                else:
                                    sentiment_html = "<span style='color:orange;font-weight:bold'>Neutral</span>"
                                st.markdown(f"Extracted/Selected Aspect: {selected_aspect}", unsafe_allow_html=True)
                                st.markdown(f"Predicted Sentiment: {sentiment_html}", unsafe_allow_html=True)
                                st.write("----------------------")
                        else:
                            st.write("No reviews found for this aspect with the selected filters.")
                else:
                    st.write("No aspect data available.")
            st.download_button(
                "Download Results as CSV",
                df_profile.to_csv(index=False),
                "results.csv",
                "text/csv"
            )
        else:
            st.error("The 'extracted_aspects' column is missing. Cannot compute dashboard metrics.")

elif main_task == "Live Inference":
    st.header("Live Inference")
    st.write("Enter a review sentence and optionally an aspect to get a predicted sentiment.")
    
    # Domain selection for Live Inference.
    domain = st.radio("Select Domain", options=["Laptop", "Restaurant"], index=0)
    
    review_input = st.text_area("Review Text", placeholder="Enter your review here...", height=100)
    aspect_input = st.text_input("Aspect (Optional)", placeholder="Enter the aspect (e.g., battery life)")
    
    if st.button("Predict Sentiment"):
        if review_input:
            with st.spinner("Running inference..."):
                # Choose endpoints based on selected domain.
                if domain == "Restaurant":
                    extraction_url = restaurant_aspect_extraction_url
                    sentiment_url = restaurant_sentiment_classification_url
                else:
                    extraction_url = laptop_aspect_extraction_url
                    sentiment_url = laptop_sentiment_classification_url
                
                # If an aspect is provided, use it; otherwise, extract aspects.
                if aspect_input.strip():
                    aspects_to_process = [aspect_input.strip()]
                else:
                    extracted_aspects = call_aspect_extraction_custom(review_input, extraction_url)
                    if extracted_aspects:
                        aspects_to_process = [aspect.strip() for aspect in extracted_aspects.split(",")]
                    else:
                        st.error("No aspect extracted. Please try again.")
                        aspects_to_process = []
                if aspects_to_process:
                    st.write("Review:", review_input)
                    for aspect in aspects_to_process:
                        sentiment = call_sentiment_classification_custom(review_input, aspect, sentiment_url)
                        if sentiment.lower() == "positive":
                            sentiment_html = "<span style='color:green;font-weight:bold'>Positive</span>"
                        elif sentiment.lower() == "negative":
                            sentiment_html = "<span style='color:red;font-weight:bold'>Negative</span>"
                        else:
                            sentiment_html = "<span style='color:orange;font-weight:bold'>Neutral</span>"
                        st.markdown(f"**Extracted/Selected Aspect:** {aspect}", unsafe_allow_html=True)
                        st.markdown(f"**Predicted Sentiment:** {sentiment_html}", unsafe_allow_html=True)
                        st.write("----------------------")
                else:
                    st.error("No aspect available for sentiment classification.")
        else:
            st.warning("Please provide review text.")
