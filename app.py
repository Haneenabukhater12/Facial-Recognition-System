
import streamlit as st
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import joblib
import numpy as np
import gdown
import os

# =============================
# 1️⃣ Download the trained model automatically from Google Drive
# =============================
MODEL_PATH = "classifier.pkl"

if not os.path.exists(MODEL_PATH):
    st.info("⬇️ Downloading model from Google Drive...")
    url = "https://drive.google.com/uc?id=1tYcltIFwMc6lx1f4z5ZkDcIpy6eETz3R"   
    gdown.download(url, MODEL_PATH, quiet=False)

# Load the model after download
classifier = joblib.load(MODEL_PATH)

# =============================
# 2️⃣ Load FaceNet and MTCNN
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# =============================
# 3️⃣ Function to extract embeddings
# =============================
def get_embedding(img):
    face = mtcnn(img)
    if face is None:
        return None
    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = facenet(face).detach().cpu().numpy()
    return embedding.reshape(1, -1)

# =============================
# 4️⃣ Streamlit Interface
# =============================
st.title("Face Recognition App")
st.write("Upload a face image or use your camera to recognize a person.")

choice = st.radio("Choose input method:", ("Upload Image", "Use Camera"))

image = None
if choice == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

elif choice == "Use Camera":
    img_file = st.camera_input("Take a picture")
    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="Captured Image", use_column_width=True)

if image is not None:
    embedding = get_embedding(image)
    if embedding is not None:
        pred = classifier.predict(embedding)[0]
        st.success(f"✅ Person recognized: {pred}")
    else:
        st.warning("❌ No face detected. Please try again.")

