import streamlit as st
from PIL import Image
import pickle as pkl

class_list = {'0': 'Negative', '1': 'Other', '2': 'Positive'}
with open("styles.css") as f:
    custom_css = f.read()
st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
st.title('Sentiment analysis from Vietnamese students’ feedback')

image = Image.open('vsfc.jpg')
st.image(image)

input_md = open('models.pkl', 'rb')
model = pkl.load(input_md)

if hasattr(model, 'predict'):
    st.header('Write a feedback')
    txt = st.text_area('', '')

    if txt != '':
        if st.button('Predict'):
            try:
                # Thực hiện dự đoán
                prediction = model.predict([txt])

                # Chuyển đổi kết quả thành string để tránh lỗi Attribute Error
                label = str(prediction[0])

                st.header('Result')
                st.text(class_list[label])
            except Exception as e:
                st.write(f"Error during prediction: {e}")
else:
    st.write("Model object does not have 'predict' method.")
