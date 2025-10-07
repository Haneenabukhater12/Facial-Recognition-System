# 🧠 Facial Recognition System

---

## 📌 Problem Statement

Traditional authentication systems — such as passwords, ID cards, or PIN codes — are vulnerable to theft, forgery, and misuse.  
This project aims to build a **deep learning-based facial recognition system** capable of identifying individuals accurately and securely, even under variations in **lighting**, **pose**, or **facial expressions**.

---

## 🎯 Project Overview

This system uses **FaceNet (InceptionResnetV1)** to extract deep facial embeddings, followed by an **SVM (Support Vector Machine)** classifier for recognition.  
It supports both image upload and live camera input through **Streamlit** or **Gradio**.

### ✅ Main Objectives:
- Achieve **high recognition accuracy**
- Ensure **low false acceptance rate (FAR)**
- Enable **real-time face recognition** deployment

---

## 🗂 Dataset

**Dataset:** *LFW (Labeled Faces in the Wild)*  
- Original dataset: ~13,000 images  
- After filtering: ~9,100 valid images  
- Distinct identities: ~1,600 people  
- Data cleaned and filtered to ensure only individuals with multiple images remain  

### 🧹 Preprocessing Steps:
- Automatic face detection using **MTCNN**
- Face alignment and cropping
- Image resizing & normalization
- Embedding extraction via **FaceNet**
- Embedding vectors stored in NumPy arrays (512 features per face)

---

## 🔍 Exploratory Data Analysis (EDA)
- Verified image size consistency after preprocessing  
- Explored number of samples per identity  
- Filtered rare identities with less than 2–3 images  
- Balanced dataset for fair training and testing  

---

## 🤖 Modeling

### 🔹 Model Architecture:
1. **Face Embedding Extraction:**  
   - Pretrained `InceptionResnetV1` (FaceNet) from `facenet-pytorch`  
   - Output: 512-dimensional embedding vector per face  

2. **Classifier:**  
   - `SVC` (Support Vector Classifier) with RBF kernel  
   - Trained on extracted embeddings

### ⚙️ Training Setup:
- Train/Test Split: 80/20  
- Optimized SVM hyperparameters  
- Model saved as: `classifier.pkl`

---

## 📊 Results

| Metric | Value |
|--------|--------|
| Accuracy | **≈ 80.0%** |
| Precision (macro avg) | 0.63 |
| Recall (macro avg) | 0.70 |
| F1-score (weighted avg) | 0.70 |

- Correctly recognized well-represented identities  
- Misclassifications mainly for identities with few samples  
- Significant accuracy improvement after using FaceNet embeddings  

---

## 🚀 Deployment

### 🧱 Platform Options:
1. **Streamlit App (Local / VS Code)**  
   Run the following command:
   ```bash
   streamlit run app.py
