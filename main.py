import os
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from PIL import Image
from src.visualizations import DataVisualizer
from src.styles import TITLE_STYLE, SIDEBAR_STYLE
from src.streamlit_utils import DataContent, DataTable

st.set_page_config(
    page_title="Signature Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)
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

def convert_df_to_csv(df):
        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue()

def main():
    st.markdown(TITLE_STYLE, unsafe_allow_html=True)
    st.markdown(SIDEBAR_STYLE, unsafe_allow_html=True)

    st.markdown('<h1 class="styled-title">Siamese Network Based Signature Verification Application</h1>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-title">Select Options</div>', unsafe_allow_html=True)
    

    if 'page' not in st.session_state:
        st.session_state['page'] = "Problem Statement"

    if "df" not in st.session_state:
        st.session_state.df = None 
    
    if 'pre_df' not in st.session_state:
        st.session_state.pre_df = None
    
    if 'data' not in st.session_state:
        st.session_state.data = None

    if 'preprocessed' not in st.session_state:
        st.session_state.preprocessed = None

    # Sidebar buttons
    if st.sidebar.button("Problem Statement"):
        st.session_state['page'] = "Problem Statement"

    if st.sidebar.button("Project Data Description"):
        st.session_state['page'] = "Project Data Description"

    if st.sidebar.button("Sample Training Data"):
        st.session_state['page'] = "Sample Training Data"

    if st.sidebar.button("Data Preprocessing"):
        st.session_state['page'] = "Data Preprocessing"

    if st.sidebar.button("Machine Learning Models Used"):
        st.session_state['page'] = "Machine Learning Models Used"

    if st.sidebar.button("Model Predictions"):
        st.session_state['page'] = "Model Predictions"

################################################################################################################

    if st.session_state['page']== "Problem Statement":
        st.image("./1_XsJ3MkPdeQ5S39skeIDU2w.jpg")
        st.markdown(DataContent.problem_statement)
    
    elif  st.session_state['page'] == "Project Data Description":
        st.markdown(DataContent.project_data_details)

    elif st.session_state['page'] == "Sample Training Data":
        st.header("📂 Sample Training Images")

        # Refresh button
        if st.button("🔄 Click here to see different Images"):
            st.rerun()

        image_dir = r"C:\Major Project\siamese_model\images"  # adjust this path

        # To display different images every time
        import random
        image_files = random.sample(
            [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))],
            k=12  # number of images to show
        )

        cols = st.columns(4)  # 4 images per row

        for i, image_file in enumerate(image_files):
            img_path = os.path.join(image_dir, image_file)
            with cols[i % 4]:  # cycle through columns
                st.image(Image.open(img_path), caption=image_file, use_container_width=True)



    elif  st.session_state['page'] == "Data Preprocessing":
        st.markdown(DataContent.Data_preprocessing)
    
    elif st.session_state['page'] == "Machine Learning Models Used":
        st.markdown(DataContent.ml_models)
    
    elif st.session_state['page'] == "Model Predictions":

        st.title("✍️ Signature Verification using Siamese Network")
        st.write("Upload two images to compare signatures using a trained Siamese Network.")

        # Upload two images
        col1, col2 = st.columns(2)

        with col1:
            image1 = st.file_uploader("Upload First Signature", type=["png", "jpg", "jpeg"])
            if image1:
                img1_display = Image.open(image1)
                st.image(img1_display, caption="First Signature", use_container_width=True)  # Large Display

        with col2:
            image2 = st.file_uploader("Upload Second Signature", type=["png", "jpg", "jpeg"])
            if image2:
                img2_display = Image.open(image2)
                st.image(img2_display, caption="Second Signature", use_container_width=True)  # Large Display

        if image1 and image2:
            # Load and preprocess images
            img1 = preprocess_image(Image.open(image1))
            img2 = preprocess_image(Image.open(image2))

            # Model Prediction (assuming 'model' is already loaded)
            prediction = model.predict([img1, img2])
            print("Raw Model Output:", prediction)

            match_probability = prediction[0][0]  # Adjust based on model output format

            # Set threshold for similarity
            THRESHOLD = 0.6  # Adjust for optimal performance

            st.subheader("🔎 Result")
            if match_probability >= THRESHOLD:
                st.success(f"✅ Signatures Match! Genuine (Similarity Score: {match_probability:.2f})")
            else:
                st.error(f"❌ Signatures Do Not Match!! Forged! (Similarity Score: {match_probability:.2f})")

        st.write("👨‍💻 **Learn more:** Siamese Networks use deep learning to compare two images and determine similarity based on feature embeddings.")


if __name__ == "__main__":
    main()

