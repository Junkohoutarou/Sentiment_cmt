import streamlit as st
from PIL import Image
import pickle as pkl
import numpy as np

class_list = {'0': 'Negative', '1': 'Neutral', '2': 'Positive'}
st.title("Sentiment analysis from Vietnamese students’ feedback")

image = Image.open('vsfc.jpg')
st.image(image)
# Đảm bảo file ec_model.pkl tồn tại
