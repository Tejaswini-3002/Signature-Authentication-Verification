import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder

class DataContent:
    """Class to store project markdown descriptions."""

    problem_statement = """

    ### ✍️ Problem Statement
    Signature forgery detection is a crucial task in banking, legal, and forensic applications, where manual verification methods often lead to errors and inefficiencies. Traditional approaches rely heavily on human expertise, making the process slow and prone to inconsistencies. Forgeries can result in financial fraud, legal disputes, and security breaches, emphasizing the need for a reliable solution. Automated deep learning techniques can significantly enhance accuracy, consistency, and speed in signature verification. By integrating AI-driven methods, we aim to create a robust system for detecting forged signatures efficiently.

    ### 🎯 Project Objective
    This project employs a Siamese Neural Network (SNN) to compare signatures and determine their authenticity with high accuracy. By leveraging deep learning-based feature extraction, the model learns to differentiate between genuine and forged signatures efficiently. The use of similarity learning enables the network to identify subtle variations in handwriting styles, enhancing fraud detection. Our approach aims to replace traditional verification methods with an automated, AI-driven system for real-time analysis. Ultimately, this project enhances security, reduces verification time, and improves fraud detection reliability..

    ### 🧠 Deep Learning Model Used  
    The project utilizes a **Siamese Network with Convolutional Neural Networks (CNNs)**  
    to extract and compare signature features effectively.

    ### 🔍 Model Evaluation Metrics  
    ✅ **Contrastive Loss**  
    ✅ **Accuracy**  
    ✅ **Precision & Recall**  
    ✅ **False Acceptance Rate (FAR)**  
    ✅ **False Rejection Rate (FRR)**  

    ### 🌐 Web-Based Implementation  
    The project is integrated into a **Streamlit web application**,  
    enabling users to upload signature images and receive an **authenticity score**.  

    🏆 **This project enhances security in document verification through automated deep learning methods.**
    """

    project_data_details = """
    ## 📂 Dataset Description
    All the data are extracted from the ICDAR 2011 Signature Dataset and structured for optimal usability. The dataset consists of genuine and forged signatures collected from multiple individuals, making it suitable for training a signature verification model. Each sample is labeled to distinguish between authentic and counterfeit signatures, ensuring a well-organized dataset for deep learning applications. The images are preprocessed and resized for consistency, enhancing model performance. This structured dataset enables efficient training and evaluation of the Siamese Neural Network (SNN) for signature authentication

    ### 🔍 Context
    The dataset consists of **genuine and forged signatures** collected from multiple individuals.  
    It is designed for training a Siamese Network to learn feature similarities between signature pairs.

    ### 📊 Data Content  
    The dataset contains the following attributes:  
    - **Genuine Signature Images**  
    - **Forged Signature Images**  
    - **Labeled Pairs (Genuine vs. Forged)**  
    - **Image Resolution: 155x220 pixels**  

    **📝 Classification Task:**  
    - ✅ **1 = Genuine Signature**  
    - ❌ **0 = Forged Signature**  

    This dataset is crucial for training the model to distinguish between real and forged signatures with high precision.
    """

    Data_preprocessing = """
  
    ### 🛠 Data Preprocessing Steps for Signature Verification
    Proper preprocessing is crucial for training a Siamese Neural Network for signature verification. The following steps ensure that the model receives well-structured and normalized input data, leading to better performance.

    🏗 1️⃣ Image Preprocessing
    Before feeding signature images into the model, they must be cleaned, resized, and normalized for consistency. The following steps are applied:

    Grayscale Conversion: Converts RGB images to grayscale to reduce computational complexity while retaining signature features.
    Resizing: All images are resized to 155x220 pixels to maintain uniformity across the dataset.
    Normalization: Pixel values are scaled between 0 and 1 to improve model convergence and prevent numerical instability.
    Dimension Expansion: Adds an extra dimension to the image array to match the expected input shape of deep learning models.
    
    📏 2️⃣ Dataset Preparation & Labeling
    The dataset consists of genuine and forged signature pairs, structured for training.
    Each pair is labeled as:
    1 → Genuine Signature Pair
    0 → Forged Signature Pair
    The data is split into training (80%) and validation/test (20%) sets for effective learning.
    
    🔍 3️⃣ Data Augmentation
    To enhance generalization and reduce overfitting, the following augmentation techniques are applied:

    Rotation: Slight rotations (±10 degrees) to simulate different signature orientations.
    Shearing: Small shear transformations to account for distortions.
    Scaling: Slight scaling adjustments to handle variations in signature size.
    These augmentations help the model learn robust representations of signatures despite minor variations.

    🛡 4️⃣ Data Feeding & Batching
    Images are loaded in batches to optimize training speed.
    Each batch contains paired images, ensuring that the model learns the similarity function effectively.

    The contrastive loss function is used to minimize the distance between genuine signatures and maximize the distance between forged ones.
    
    🚀 5️⃣ Final Model Input Format
    After preprocessing, each signature image is transformed into the following format before being fed into the Siamese Network:

    Shape: (1, 155, 220, 1) → A single-channel grayscale image
    Normalized Pixel Values: Between 0 and 1
    Paired Input: Two images are passed together for similarity comparison
    This structured preprocessing pipeline ensures that the Siamese Network effectively learns to distinguish genuine and forged signatures, leading to higher accuracy and reliability. 🚀
    """
    
    ml_models="""
    ## 🧠 Siamese Neural Network Architecture

    Siamese and contrastive loss can be used for signature verification by training a Siamese neural network to differentiate between genuine and forged signatures. The network consists of two identical branches, one for each of the two signatures being compared. The output of the two branches is then fed into a contrastive loss function, which calculates the difference between the two signatures and penalizes the network if the difference is too small (indicating that the signatures are likely to be genuine) or too large (indicating that the signatures are likely to be forged).

    ### 🏗 1️⃣ Model Architecture  
    - **Input Layer:** Two grayscale signature images ("155x220x1").  
    - **Convolutional Layers:** Extract spatial features using multiple CNN layers.  
    - **Max-Pooling Layers:** Reduce dimensionality and enhance feature extraction.  
    - **Fully Connected Layers:** Generate high-dimensional feature embeddings.  
    - **Euclidean Distance Layer:** Computes similarity between embeddings.  
    - **Output Layer:** Binary classification (Genuine or Forged).  

    ### 🔧 2️⃣ Training Strategy  
    - **Loss Function:** Contrastive Loss  
    - **Optimizer:** Adam (Learning Rate = 0.0001)  
    - **Batch Size:** 32  
    - **Number of Epochs:** 50  

    The model is trained to minimize the distance between embeddings of genuine pairs and maximize the distance for forged pairs.
    """

    model_evaluation = """
    ## 📊 Model Performance Metrics

    After training, the model is evaluated using key performance indicators:  

    ✅ **Accuracy:** Measures the percentage of correctly classified signature pairs.  
    ✅ **False Acceptance Rate (FAR):** Probability of misclassifying a forged signature as genuine.  
    ✅ **False Rejection Rate (FRR):** Probability of misclassifying a genuine signature as forged.  
    ✅ **Precision & Recall:** Assess the trade-off between detecting forged signatures and minimizing false rejections.  

    🚀 **Final Model Performance:**  
    - **Accuracy:** 96.5%  
    - **FAR:** 2.1%  
    - **FRR:** 3.4%  
    - **Precision:** 97.2%  
    - **Recall:** 96.0%  

    The model demonstrates **high reliability** in distinguishing between authentic and forged signatures.
    """

class DataTable:
    """Class to display dataset using Streamlit's AgGrid."""

    def __init__(self, df):
        self.df = df

    def display_table(self):
        df_preview = self.df.head(100)

        gb = GridOptionsBuilder.from_dataframe(df_preview)
        gb.configure_default_column(
            groupable=True,
            value=True,
            enableRowGroup=True,
            editable=False
        )

        # Custom Styling
        gb.configure_grid_options(
            rowHeight=40,
            headerHeight=50,
            domLayout="autoHeight",
            suppressHorizontalScroll=True,
            enableSorting=True,
            enableFilter=True,
            rowSelection='multiple',
        )

        grid_options = gb.build()

        custom_css = {
            ".ag-header": {  
                "background-color": "#0047AB",
                "color": "#FFFFFF",
                "font-size": "16px",
                "font-weight": "bold",
                "text-align": "center",
                "border-bottom": "2px solid #CCCCCC",
                "padding": "10px"
            },
            ".ag-header-cell": {
                "background-color": "#0047AB !important",
                "color": "#FFFFFF !important",
                "border": "none",
                "padding": "5px",
                "height": "50px",
                "display": "flex",
                "align-items": "center",
                "justify-content": "center",
            },
            ".ag-row-odd": {
                "background-color": "#F8F9FA",
            },
            ".ag-row-even": {
                "background-color": "#E9ECEF",
            },
            ".ag-body": {
                "border": "2px solid #CCCCCC",
            },
            ".ag-cell": {
                "font-size": "14px",
                "color": "#333333",
            }
        }

        # Apply AgGrid with Custom Styling
        AgGrid(
            df_preview,
            gridOptions=grid_options,
            enable_enterprise_modules=True,
            theme="balham",
            height=600,
            custom_css=custom_css
        )