import keras
from keras import models
from keras.preprocessing.image import img_to_array
import streamlit as st
from PIL import Image
import numpy as np
import cv2

model = keras.models.load_model('SLFModel.pb')
model = keras.Sequential([
  model,
  keras.layers.Softmax()
])
class_names = ['egg', 'no egg']

st.write("SLF Resnet Model Created Through Transfer Learning")
imgFile = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])

def import_and_predict(model, image):
  #image = keras.applications.resnet50.preprocess_input(image)
  image = cv2.resize(image,(224,224))     # resize image to match model's expected sizing
  image = image.reshape(1,224,224,3)
  prediction = model.predict(image)
  st.text(prediction)
  return prediction

if imgFile is None:
  st.text("Please upload an image to classify")
else:
  image = Image.open(imgFile)
  image_array = img_to_array(image)
  image_array_test = image_array / 255
  st.image(image_array_test, use_column_width = True)
  prediction = import_and_predict(model, image_array)
  category = np.argmax(prediction)
  object_type = class_names[category]
  st.write("The given image is of category " + object_type)
