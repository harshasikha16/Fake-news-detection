import streamlit as st
import joblib

# Load the trained model and vectorizer
svm_model = joblib.load("best_fake_news_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Function to predict Fake/Real news
def predict_news(news_text):
    news_tfidf = tfidf_vectorizer.transform([news_text])
    prediction = svm_model.predict(news_tfidf)
    return "📰 Real News ✅" if prediction[0] == 1 else "🚨 Fake News ❌"

# Streamlit Web App
st.title("📰 Fake News Detection App")
st.write("Enter a news article to check if it is real or fake.")

# User input box
user_input = st.text_area("Paste your news article here:")

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a news article.")
    else:
        result = predict_news(user_input)
        st.success(f"Prediction: {result}")
