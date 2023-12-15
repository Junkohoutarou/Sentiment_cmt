import streamlit as st
from PIL import Image
import pickle as pkl

class_list = {'0': 'Negative', '1': 'Neutral', '2': 'Positive'}
with open("styles.css") as f:
    custom_css = f.read()
st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
st.title('Sentiment analysis from Vietnamese students’ feedback')

image = Image.open('vsfc.jpg')
st.image(image)

try:
    with open('ec_model.pkl', 'rb') as input_ec:
        encoder = pkl.load(input_ec)
except FileNotFoundError:
    st.error("Error: 'ec_model.pkl' not found. Please check the file path.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading 'ec_model.pkl': {e}")
    st.stop()

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
    if st.button('Predict'):
        feature_vector = encoder.transform([txt])
        label = str((model.predict(feature_vector))[0])

        st.header('Result')
        st.text(class_list[label])
