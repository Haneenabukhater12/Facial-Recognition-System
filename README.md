# 🧠 Facial Recognition System  

A deep learning–based facial recognition pipeline that combines **FaceNet embeddings** with an **SVM classifier** for robust and accurate identity recognition.  
This system provides both a research-grade model and an interactive user interface built with **Gradio**.

---

## 📌 Problem Statement  

Traditional authentication methods like passwords or ID cards are often insecure and vulnerable to theft, forgery, or loss.  
This project aims to build a **face recognition system** that identifies individuals *reliably and securely*, even under changes in lighting, pose, or facial expression.

---

## 🎯 Project Overview  

The system leverages a **pretrained FaceNet (InceptionResnetV1)** network to extract deep feature embeddings from face images.  
These embeddings are then classified using a **Support Vector Machine (SVM)** model trained on labeled facial features.  

✅ **Key Features:**  
- High-accuracy face recognition  
- Works with image uploads or live camera input  
- Lightweight and deployable via Streamlit or Gradio  
- Transfer learning from a pretrained VGGFace2 model  

---

## 🗂 Dataset  

**Dataset Used:** [LFW (Labeled Faces in the Wild)](https://www.kaggle.com/datasets/wsygina/lfw-funneled)  
- Total Images: ~13,000  
- Cleaned & filtered: ~9,100  
- Unique identities: ~1,600  
- Only identities with ≥2 images retained  

**Preprocessing Pipeline:**  
1. Automatic face detection using **MTCNN**  
2. Face alignment and cropping  
3. Image resizing and normalization  
4. Embedding extraction using **FaceNet (512 features per face)**  
5. Features stored in NumPy arrays for efficient training  

---

## 🔍 Exploratory Data Analysis (EDA)  

- Ensured consistent image dimensions post-processing  
- Analyzed distribution of images per identity  
- Removed underrepresented classes (≤2 samples)  
- Visualized class balance and embedding space distribution  

---

## 🤖 Modeling  

### 🔹 Architecture  

| Component | Description |
|------------|-------------|
| **FaceNet (InceptionResnetV1)** | Extracts 512-dimensional face embeddings |
| **SVM Classifier (RBF Kernel)** | Performs classification based on embeddings |

### ⚙ Training Details  
- Split: 80% training / 20% testing  
- Optimized hyperparameters for SVM  
- Model exported as: `classifier.pkl`

---

## 📊 Results  

| Metric | Value |
|--------|--------|
| Accuracy | **≈ 80%** |
| Precision (macro avg) | 0.63 |
| Recall (macro avg) | 0.70 |
| F1-score (weighted avg) | 0.70 |

🟢 Strong recognition performance for well-represented identities  
🔴 Some misclassifications for individuals with limited samples  

---

## 💻 Implementation  

You can view the full training and preprocessing pipeline in the notebook below:  
📓 **Notebook:** [`Untitled11.ipynb`](./Untitled11.ipynb)

---

## 🚀 Deployment  

### 🧩 Gradio Web App  

Try the live demo here:  
👉 **[Facial Recognition Web App](https://01185056159e72bdea.gradio.live/)**  

#### Run Locally
```bash
pip install gradio torch torchvision facenet-pytorch pillow scikit-learn joblib numpy
python app.py
