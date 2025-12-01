# Signature Authentication & Verification using Siamese Networks
This project implements a deep-learning based system to authenticate handwritten signatures using a **Siamese Neural Network**. The model learns to compare two signatures and determine whether they belong to the same person (genuine) or are forged.

## Features
- Siamese Neural Network architecture for similarity learning  
- Signature image preprocessing  
- Pair generation (genuine & forged pairs)  
- Training the siamese network 
- Contrastive loss for distance-based learning  
- Signature verification using Euclidean distance  
- Model evaluation with accuracy and loss graphs
  
## How It Works
1. Two signature images are passed into identical CNN branches.  
2. The Siamese model extracts feature embeddings.  
3. A distance metric (Euclidean distance) calculates similarity.  
4. If the distance is small then signatures match (genuine).  
5. If the distance is large yjen the signatures do not match (forged).

## Technologies Used
- Pandas 
- OpenCV  
- NumPy  
- Matplotlib


