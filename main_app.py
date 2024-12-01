import numpy as np 
import streamlit as st 
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

model = tf.keras.models.load_model("D:\Dog_breed\dog_breed.h5")

CLASS_NAMES = ['Scottish Deerhound','Maltese Dog','Bernese Mountain Dog']

st.title("Dog Breed Prediction")
st.markdown("Upload an image of the dog")

dog_image = st.file_uploader("Choose an image..",type="png")
submit = st.button("Predict")

if submit:
    file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    # Displaying the image
    st.image(opencv_image, channels="BGR")
    #Resizing the image
    opencv_image = cv2.resize(opencv_image, (224,224))
    #Convert image to 4 Dimension
    opencv_image.shape = (1,224,224,3)
    #Make Prediction
    Y_pred = model.predict(opencv_image)
    st.title("The Dog Breed is: " + CLASS_NAMES[np.argmax(Y_pred)])  