
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2

model = load_model('model')

canvas_result = st_canvas(
    stroke_width=10,
    stroke_color='#ffffff',
    background_color='#000000',
    height=150,width=150,
    drawing_mode="freedraw"
)

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    #st.image(canvas_result.image_data)

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val = model.predict(test_x.reshape(1,28, 28,1))
    st.write(f'result: {np.argmax(val[0])}')
