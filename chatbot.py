import streamlit as st 
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
genai.configure(api_key="AIzaSyAHTsuUjUOaZk-9xWvEROaGToB5lLh2On4")
model = genai.GenerativeModel("gemini-2.5-flash")

st.set_page_config(page_title="Gemini Chatbot for Ml Task")
st.title("Gemini Chatbot for ML Task")

user_input = st.text_area("Enter your question or request:", height=150)
uploaded_image = st.file_uploader("Upload an image (optional):", type=["png", "jpg", "jpeg"])

if st.button("Send"):
    inputs = []

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Your uploaded image", use_column_width=True)
        inputs.append(image) 

    if user_input:
        inputs.append(user_input)

    if inputs:
        st.write("thinking")  
        response = model.generate_content(inputs) 
        st.subheader("Answer")

        st.write(response.text) 