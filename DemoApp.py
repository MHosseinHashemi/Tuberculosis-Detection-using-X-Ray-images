from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow import keras
import streamlit as st
from PIL import Image
import cv2
import numpy as np


# Load model
my_model = load_model("EfficientNetV2S.h5")

# UI Design
st.set_page_config(layout='wide')
st.sidebar.markdown("<div><img src='https://cdn1.poz.com/69152_lungs-ts-147323597.jpg_1dff2524-28f8-4ce0-864e-e1c9c4e71a25.jpeg' width=100 /></div>", unsafe_allow_html=True)
st.sidebar.title("Tuberculosis (TB) is a disease caused by germs that are spread from person to person through the air. This web service aims to assist the physicians to analyze the patients chest x-ray images for any abnormallies relavant to TB.")
st.sidebar.markdown('')
left_col, center_col, right_col = st.columns(3) 


# a function to choose the right label to show
def labelizer(p):
    if p==0:
        return 'The case is Normal'
    elif p==1:
        return 'The case is Tuberclosis'




# a function to manipulate the input pic
def classifier(image, model):
    image = np.array(image)
    image = cv2.resize(image, (512,512))
    image = image.reshape(1,512,512,3)
    # predicting the label
    prediction = model.predict_on_batch(image)
    # map the predition to labels
    classification = np.where(prediction == np.max(prediction))[1][0]
    output = "With " + str(int(prediction[0][classification]*100)) + "% Confidence, " + labelizer(classification)

    return output


   

input_file = st.file_uploader("Upload MRI pic", type=['jpg','png'])
if input_file is None:
    st.text("Please upload a picture")
else:
    img = Image.open(input_file)
    with center_col:
        st.image(img, use_column_width=True, caption="Your uploaded file")
        pred = classifier(img, my_model)
    st.success(pred)
    

