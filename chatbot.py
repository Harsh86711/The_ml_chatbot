import streamlit as st
import pandas as pd
import pickle
import nltk
import string
import os
import random

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
# Page Configuration
st.set_page_config(
    page_title="ML Chatbot",
    page_icon="💀",
    layout="wide"
)

# Base Directory Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# NLTK Preprocessing Setup
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    clean_tokens = [
        stemmer.stem(word) 
        for word in tokens 
        if word not in string.punctuation and word not in stop_words
    ]
    return " ".join(clean_tokens)

# Load Model Files
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "Chatbot_model.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

# Load Dataset for Responses
@st.cache_data
def load_data():
    csv_path = os.path.join(BASE_DIR, "chatbot.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

try:
    model, vectorizer = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Error loading model or files: {e}")
    st.stop()

# Custom Responses Dictionary (Fallback)
RESPONSES = {
    "greeting": ["Hello! How can I help you?", "Hi there! How can I assist you today?"],
    "goodbye": ["Goodbye! Have a nice day!", "See you later!"],
    "thanks": ["You're welcome!", "Happy to help!"],
    "about": ["I am an ML Chatbot trained to answer queries about Python, AI, ML, Deep Learning, and Data Science."]
}

def get_response(tag):
    # First check CSV if available
    if df is not None:
        cols = [c.lower() for c in df.columns]
        if 'tag' in cols and 'response' in cols:
            tag_col = df.columns[cols.index('tag')]
            resp_col = df.columns[cols.index('response')]
            matching = df[df[tag_col] == tag][resp_col].values
            if len(matching) > 0:
                return random.choice(matching)
    
    # Fallback to dictionary
    if tag in RESPONSES:
        return random.choice(RESPONSES[tag])
    
    # Default fallback message seen in your screenshot
    return "Sorry, I am not sure about that. Please ask something related to AI, ML, Python, or Data Science."

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("About Chatbot")
    st.write("This chatbot uses:")
    st.markdown("- 🟩 **NLP Text Processing**")
    st.markdown("- 🟩 **TF-IDF Vectorization**")
    st.markdown("- 🟩 **Logistic Regression**")
    st.markdown("- 🟩 **Streamlit**")
    
    st.write("")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ----------------- MAIN UI -----------------
st.markdown("<h1 style='text-align: center;'>💀 Machine Learning Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Ask questions about Python, AI, Machine Learning, Deep Learning and Data Science.</p>", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if user_input := st.chat_input("Type your message here..."):
    # Render user input
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate & Render bot response
    with st.chat_message("assistant"):
        processed = preprocess_text(user_input)
        vectorized = vectorizer.transform([processed])
        predicted_tag = model.predict(vectorized)[0]
        
        reply = get_response(predicted_tag)
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})