import streamlit as st
from transformers import pipeline
import nltk
import os

# Set the NLTK data path to the local directory
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_dir)

# Function to summarize text using a transformer model
def summarize_text(text, max_length=200, min_length=50):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    summary_text = " ".join([summary['summary_text'] for summary in summaries])
    return summary_text.split('. ')

# Streamlit app
st.title("Text Summarizer")

# Input text
input_text = st.text_area("Enter the text you want to summarize")

if input_text:
    try:
        # Summarize the input text
        summary_points = summarize_text(input_text)

        # Display the summary
        st.subheader("Summary")
        for i, point in enumerate(summary_points, 1):
            st.write(f"{i}. {point}")

    except Exception as e:
        st.error(f"An error occurred: {e}")