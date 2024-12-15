import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Function to load the trained model
def load_model():
    model = tf.keras.models.load_model('C:\\Users\\Ahmed Sarhan\\Downloads\\my_model.h5')  # Replace with your model path
    return model

# Function to preprocess the uploaded image
def load_and_prep_image(uploaded_file, img_shape=224):
    img = image.load_img(uploaded_file, target_size=(img_shape, img_shape))  # Resize image
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = img_array / 255.0  # Normalize image (same as model training)
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Main function to run the Streamlit app
def main():
    st.title("Image Classification with Pretrained Model")

    # Upload model
    model = load_model()

    # File uploader to upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display the image to the user
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        img = load_and_prep_image(uploaded_file)
        
        # Make prediction
        pred = model.predict(img)
        
        # Display prediction result
        if pred <= 0.5:
            st.write("Prediction: Infected (Class 1)")
        else:
            st.write("Prediction: Non-Infected (Class 0)")

# Run the Streamlit app
if __name__ == "__main__":
    main()
