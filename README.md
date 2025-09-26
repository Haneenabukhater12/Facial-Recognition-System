 # Facial Recognition System

---

## 📌 Problem Statement

Traditional security systems (passwords, ID cards, PINs) are vulnerable to theft, forgery, and misuse.  
This project develops a facial recognition system that can authenticate or identify individuals reliably, even under variations in lighting, pose, or facial expressions.

---

## 🎯 Project Overview

Build a face recognition model with:

- High accuracy
- Low false acceptance rate (FAR)
- Real-time deployment for testing

---

## 🗂 Dataset

*LFW (Labeled Faces in the Wild)*

- Total images: 3023
- Classes: 7 identities (e.g., Vladimir Putin, Tony Blair, Tiger Woods, etc.)

*Preprocessing:*

- Resized & normalized images
- Face cropping ensured consistency
- Features extracted → flattened into vectors

---

## 🔍 Exploratory Data Analysis (EDA)

- Image size distribution → consistent at (50 × 37) pixels
- Color distribution analyzed across RGB channels
- Entropy distribution plotted to measure information content
- Data augmentation applied (flipping, rotation, scaling)

---

## 🤖 Modeling

*Baseline model:* Support Vector Classifier (SVC, linear kernel)  
*Training/Testing split:* 80/20  
*Evaluation metrics:* Accuracy, Precision, Recall, F1-score, FAR

---

## 📊 Results

- *Model Accuracy:* ~60.33%
- *Classification Report:*
  - Some classes with more samples → high precision & recall
  - Other classes with fewer samples → lower recall
  - Confusion mainly between visually similar faces
- *Confusion Matrix:* Visualized with heatmap

*Key Observations:*

- Identities with many samples achieved higher accuracy.
- Classes with limited data suffered from low recall.
- Data augmentation improved robustness against lighting & pose variation.

---

## 🚀 Deployment

*Platform:* Streamlit web app  

*Features:*

- Upload or capture an image for verification
- Real-time identity testing

---
## 🔮 Future Work

- Train deeper models (CNN, FaceNet, VGG-Face) for higher accuracy
- Expand dataset (e.g., VGGFace2) for better generalization
- Optimize performance for real-time large-scale deployment
-  ---
  *Presentation Slides:* 
 https://drive.google.com/drive/folders/1BGlp9aN-vVxYzPq_kVk6MjJZwE-7cu8x

 

## 📦 Repository Structure
