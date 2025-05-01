import streamlit as st
import joblib
import re

# Load the model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Clean the input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

# Streamlit UI
st.title("Twitter Sentiment Analysis App")
st.write("Enter a tweet and find out whether it's positive, negative, or neutral.")

user_input = st.text_area("Enter your tweet:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0]
        st.success(f"Predicted Sentiment: **{prediction.capitalize()}**")
