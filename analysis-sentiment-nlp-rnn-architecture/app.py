import streamlit as st
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title("Analysis of Sentiment in Scientific News Articles")

# Load model & tokenizer dengan path folder baru
@st.cache_resource
def load_my_model():
    model = load_model('analysis-sentiment-nlp/model_lstm.h5')
    with open('analysis-sentiment-nlp/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_my_model()

text = st.text_area("Masukkan abstrak artikel ilmiah di sini:")
if st.button("Prediksi"):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=150)
    prediction = model.predict(padded)
    
    if prediction >= 0.5:
        st.success(f"NLP Modern (Transformer/Attention-based) - Confidence: {prediction:.2%}")
    else:
        st.info(f"NLP Tradisional (RNN/LSTM) - Confidence: {(1 - prediction):.2%}")
