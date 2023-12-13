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

input_md = open('models.pkl', 'rb')
model = pkl.load(input_md)

st.header('Write a feedback')
txt = st.text_area('', '')

if txt != '':
    if st.button('Predict'):
        # Chuyển đổi phản hồi thành vector đặc trưng (nếu cần)
        # feature_vector = encode_feedback(txt)
        
        # Dự đoán tâm trạng bằng cách sử dụng chỉ model
        label = str((model.predict([txt]))[0])

        st.header('Result')
        st.text(class_list[label])
