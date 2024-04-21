#Importing Dependencies
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Configures the Streamlit page with a title and an emoji icon.
st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🖼️")

# Displays a main title 'CIFAR-10 Image Classifier' on the app page.
st.title('CIFAR-10 Image Classifier')
st.title('Author: Aaron Tekle') #my name

# Brief introduction to the web application, explaining its purpose
# and the dataset it's trained on. This markdown section informs users about the app's functionality.
st.markdown("""
This web application uses my Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset for image classification. The dataset comprises 60,000 32x32 color images across 10 categories, including vehicles and animals. Try it out by uploading an image!
""")

# Offers guidance to users on the type of images that yield the best results,
# emphasizing the relevance to the CIFAR-10 dataset classes.
st.markdown("""
READ/NOTE: (If you want the yield the best results, upload images that are related to the CIFAR-10 Dataset (in other words, upload any images that relate to Airplanes, Automobiles, Birds, Cats, Deers, Dogs, Frogs, Horses, Ships, & Trucks!)
""")

st.markdown("""
Special thanks to Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton for their work in creating the CIFAR-10 dataset.

Citation: "Krizhevsky, A., Nair, V., and Hinton, G. (2009). CIFAR-10 (Canadian Institute for Advanced Research)".
""")

# Loads the CNN model trained on the CIFAR-10 dataset.
model = load_model("my_model.h5")

# A list defining the class names corresponding to the CIFAR-10 dataset's classes.
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Creation of a file uploader widget allowing users to upload images for classification.
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"]) #file types allowed
if uploaded_file is not None:  # Checks if a file has been uploaded.
    img = Image.open(uploaded_file)  # Opens the uploaded image file.
    st.image(img, caption='Uploaded Image', use_column_width=True)  # Displays the uploaded image.
    
    img = img.resize((32, 32))  # Resizes the image to 32x32 pixels, matching the model's expected input.
    img_array = np.array(img) / 255.0  # Converts the image to an array and normalizes pixel values.
    img_array = np.expand_dims(img_array, axis=0)  # Adds a batch dimension to the array for model prediction.
    
    predictions = model.predict(img_array)  # Predicts the class of the uploaded image.
    predicted_class = class_names[np.argmax(predictions)]  # Finds the class name with the highest prediction score.
    
    st.subheader(f"Prediction: {predicted_class}")  # Displays the predicted class to the user.
