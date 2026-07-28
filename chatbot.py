import streamlit as st
import pandas as pd
import pickle
import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK Data
nltk.download('punkt')
nltk.download('stopwords')

# Page Configuration

st.set_page_config(
    page_title="ML Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Load Model Files

@st.cache_resource
def load_model():

    model = pickle.load(
        open("chatbot_model.pkl", "rb")
    )

    vectorizer = pickle.load(
        open("vectorizer.pkl", "rb")
    )

    return model, vectorizer



model, vectorizer = load_model()

# Load Dataset

@st.cache_data
def load_data():

    df = pd.read_csv("chatbot.csv")

    return df



df = load_data()

# Text Preprocessing

stemmer = PorterStemmer()

stop_words = set(
    stopwords.words("english")
)


def clean_text(text):

    # lowercase
    text = text.lower()


    # remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )


    # tokenize
    words = word_tokenize(text)


    # remove stop words + stemming
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]


    return " ".join(words)


# Chatbot Prediction Function

def chatbot_response(user_text):


    cleaned = clean_text(user_text)


    # Convert text into numbers

    vector = vectorizer.transform(
        [cleaned]
    )


    # Prediction

    intent = model.predict(vector)[0]


    # Confidence score

    probability = model.predict_proba(vector)

    confidence = probability.max()



    # Low confidence handling

    if confidence < 0.40:

        return (
            "Sorry, I am not sure about that. "
            "Please ask something related to AI, ML, Python, or Data Science."
        )



    # Get response from CSV

    response = df[
        df["intent"] == intent
    ]["response"].iloc[0]


    return response



# Session Chat History

if "messages" not in st.session_state:

    st.session_state.messages = []



# User Interface


st.title("🤖 Machine Learning Chatbot")

st.write(
    "Ask questions about Python, AI, Machine Learning, Deep Learning and Data Science."
)



# Display old messages

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )



# User Input

user_input = st.chat_input(
    "Type your message here..."
)



if user_input:


    # Display user message

    st.session_state.messages.append(
        {
            "role":"user",
            "content":user_input
        }
    )


    with st.chat_message("user"):

        st.write(user_input)



    # Get bot response

    answer = chatbot_response(
        user_input
    )


    # Display bot message

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )


    with st.chat_message("assistant"):

        st.write(answer)


# Sidebar

with st.sidebar:


    st.header("About Chatbot")


    st.write(
        """
        This chatbot uses:

        ✅ NLP Text Processing

        ✅ TF-IDF Vectorization

        ✅ Logistic Regression

        ✅ Streamlit

        """
    )


    if st.button("Clear Chat"):

        st.session_state.messages = []

        st.rerun()