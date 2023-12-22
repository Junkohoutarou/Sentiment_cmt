
import streamlit as st
from PIL import Image
import pickle as pkl
import numpy as np
class_list = {'0': 'Negative', '1': 'Neutral', '2': 'Positive'}
st.title("Sentiment analysis from Vietnamese students’ feedback")

image = Image.open('vsfc.jpg')
st.image(image)
# Đảm bảo file ec_model.pkl tồn tại
try:
    with open('ec_model.pkl', 'rb') as input_ec:
        encoder = pkl.load(input_ec)
except FileNotFoundError:
    st.error("Error: 'ec_model.pkl' not found. Please check the file path.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading 'ec_model.pkl': {e}")
    st.stop()
# Đảm bảo file model.pkl tồn tại
try:
    with open('model.pkl', 'rb') as input_md:
        model = pkl.load(input_md)
except FileNotFoundError:
    st.error("Error: 'model.pkl' not found. Please check the file path.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading 'model.pkl': {e}")
    st.stop()

st.header('Write a feedback')
txt = st.text_area('', '')

if txt != '':
    feature_vector = encoder.transform([txt])
    prediction_result = model.predict(feature_vector)

    st.header('Result')
    st.text(class_list[str(prediction_result[0])])



