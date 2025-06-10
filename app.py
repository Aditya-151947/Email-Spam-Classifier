import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))


st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("📩 Enter your message here:", height=150)

# Sidebar
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #000000;
            color: white;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)
st.sidebar.header("🛠 Features")
st.sidebar.write("🔍 Spam Detection using ML")
st.sidebar.write("📊 Trained with TF-IDF + SVM")

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            width: 250px !important;
        }
    </style>
""", unsafe_allow_html=True)


if st.button("🚀 Predict Spam Status"):

    # 1.preprocess
    transformed_sms = transform_text(input_sms)
    #2. vectorize
    vector_input = tfidf.transform([transformed_sms]).toarray() #
    #3. predict
    result = model.predict(vector_input)[0]
    #4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


# Footer
st.markdown("""
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            color: black;
        }
    </style>
    <div class='footer'>© 2025 cls | All rights reserved</div>
""", unsafe_allow_html=True)