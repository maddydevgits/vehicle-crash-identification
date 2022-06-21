import cv2
from detection import AccidentDetectionModel
import numpy as np
import os
import streamlit as st
from PIL import Image
def load_image(img):
    return Image.open(img)

st.title('Vehicle Accident Identificaiton System')
model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

src_file=st.file_uploader('Upload Image',type=['png','jpg','jpeg'])

if src_file is not None:
    st.image(load_image(src_file),width=250)
    with open('src.jpg','wb') as f:
        f.write(src_file.getbuffer())
    
    frame = cv2.imread('src.jpg')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(gray_frame, (250, 250))

    pred, prob = model.predict_accident(roi[np.newaxis, :, :])
    st.success(pred)
    if(pred == "Accident"):
        prob = (round(prob[0][0]*100, 2))
        st.write(prob)    
            # to beep when alert:
            # if(prob > 90):
            #     os.system("say beep")

