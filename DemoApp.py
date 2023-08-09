import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import *
import streamlit as st
from PIL import Image
import cv2
import numpy as np

# Load model
my_model = load_model("MobileNetV2_2.h5", compile=False)

# UI Design
st.set_page_config(layout='wide')

st.sidebar.markdown("<p style='text-align: center; font-family: Georgia, sans-serif; font-size: 42px; color: #FF5733; text-shadow: 7px 7px 7px rgba(0.0,0.0,0.0,0.2);'>TB Detector</p>", unsafe_allow_html=True)
st.sidebar.markdown("<div><img src='https://64.media.tumblr.com/bca36272f65a8dbf0575fc4ed4440b9e/e511f9264f7ec49d-32/s640x960/aa3e63c8f4f3f4abcf149ad8c7bf333bf8def188.gif' width=300 /></div>", unsafe_allow_html=True)


input_file = st.sidebar.file_uploader("", type=['jpg','png'])

st.write("<p style='text-align: center; font-size: 30px; font-family: Georgia;'>How Can I help You?</p>", unsafe_allow_html=True)
st.write("<p style='text-align: center; text-align: justify; font-size: 16px;'> <span style='font-size: 20px'><b>Tuberculosis</b></span> (TB) is an infectious disease that primarily affects the lungs and can spread from person to person through the air. A vital diagnostic approach for identifying and analyzing this condition involves the examination of Chest X-rays. This web service is designed to aid medical professionals in scrutinizing patients' chest X-rays, facilitating the detection of any pertinent abnormalities related to TB.</p><br>", unsafe_allow_html=True)
st.video("https://www.youtube.com/watch?v=UKV8Zn7x0wM")


left_col, center_col, right_col = st.columns(3) 


# a function to choose the right label to show
def labelizer(p):
    if p==0:
        return 'The case is Normal'
    elif p==1:
        return 'The case is Tuberclosis'
        

def classifier(image, model):
    # Preprocess the image
    image = image.resize((512, 512))  
    image = np.array(image) / 255.0   

    # Predicting the label
    prediction = model.predict_on_batch(np.expand_dims(image, axis=0))

    # Map the prediction to labels
    if prediction[0][0] >=0.5:
        classification = 1
        confidence = np.max(prediction[0][0]) * 100
    else:
        classification = 0
        confidence = 1 - np.max(prediction[0][0]) * 100
        
    

    output = labelizer(classification) + " Confidence: " + str(confidence) + "%"
    return output



# input_file = st.file_uploader("", type=['jpg','png'])
if input_file is None:
    pass
else:
    img = Image.open(input_file)
    with center_col:
        # st.image(img, use_column_width=True, caption="Your uploaded file")
        pred = classifier(img, my_model)
    st.sidebar.success(pred)
