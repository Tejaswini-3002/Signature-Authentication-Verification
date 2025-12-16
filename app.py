import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from PIL import Image

# Define Euclidean Distance function (needed to load the model)
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

# Load the trained Siamese model
model = load_model("siamese_model.h5", custom_objects={"euclidean_distance": euclidean_distance})

# Image Preprocessing Function
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((100, 100))  # Resize
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=(0, -1))  # Add batch and channel dimensions
    return image

# Streamlit UI
st.title("🔍 Siamese Network for Signature Verification")
st.write("Upload two images to compare signatures using a trained Siamese Network.")

# Upload two images
col1, col2 = st.columns(2)
with col1:
    image1 = st.file_uploader("Upload First Signature", type=["png", "jpg", "jpeg"])
with col2:
    image2 = st.file_uploader("Upload Second Signature", type=["png", "jpg", "jpeg"])

if image1 and image2:
    # Load and preprocess images
    img1 = preprocess_image(Image.open(image1))
    img2 = preprocess_image(Image.open(image2))

    # Display uploaded images
    st.image([image1, image2], caption=["Signature 1", "Signature 2"], width=150)

    prediction = model.predict([img1, img2])
    print("Raw Model Output:", prediction)
    match_probability = prediction[0][1]  # Assuming second output is for "same"

    # Adjust logic based on how model was trained
    match_probability = prediction[0][0]  # Using the second value (probability of being "same")

    # Adjust threshold (tune if needed)
    THRESHOLD = 0.6  # Increase slightly from 0.5 to be more forgiving

    st.subheader("🔎 Result")
    if match_probability >= THRESHOLD:  # Higher probability means same
        st.success(f"✅ Signatures Match! (Similarity Score: {match_probability:.2f})")
    else:
        st.error(f"❌ Signatures Do Not Match (Similarity Score: {match_probability:.2f})")



st.write("👨‍💻 **Learn more:** Siamese Networks use deep learning to compare two images and determine similarity based on feature embeddings.")
