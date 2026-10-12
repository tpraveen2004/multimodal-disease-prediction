import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import json
import matplotlib.pyplot as plt

from models.multimodal_model import MultimodalModel
from explainability.gradcam import generate_gradcam
from audit_logger import log_prediction

# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = "saved_models/multimodal_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_model():
    model = MultimodalModel().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    try:
        model.load_state_dict(checkpoint["model_state"])
    except:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

model = load_model()
# -----------------------
# TRANSFORM
# -----------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# -----------------------
# LABEL MAP
# -----------------------
with open("disease_labels.json", "r") as f:
    label_map = json.load(f)

inv_label_map = {v: k for k, v in label_map.items()}

# -----------------------
# FAIRNESS FUNCTION
# -----------------------
def check_fairness(model, img_tensor, ehr_tensor, device):
    ehr_male = ehr_tensor.clone()
    ehr_female = ehr_tensor.clone()

    # Change gender only
    ehr_male[0][0] = 1
    ehr_female[0][0] = 0

    with torch.no_grad():
        out_male, _, _, _ = model(img_tensor, ehr_male)
        out_female, _, _, _ = model(img_tensor, ehr_female)

    prob_male = torch.sigmoid(out_male).cpu().numpy()[0]
    prob_female = torch.sigmoid(out_female).cpu().numpy()[0]

    diff = abs(prob_male - prob_female).mean()
    return diff

# -----------------------
# UI
# -----------------------
st.title(" Multimodal Disease Prediction with Responsible AI")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])

sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=120, value=30)
survival = st.selectbox("Survival", ["alive", "dead"])
intubated = st.selectbox("Intubated", ["no", "yes"])

if st.button("Predict"):

    if uploaded_file is None:
        st.error("Please upload an image")
    else:
        # -----------------------
        # PREPARE INPUT
        # -----------------------
        img = Image.open(uploaded_file).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        ehr_tensor = torch.tensor([[
            1 if sex == "male" else 0,
            age,
            1 if survival == "dead" else 0,
            1 if intubated == "yes" else 0
        ]], dtype=torch.float32).to(DEVICE)

        # -----------------------
        # PREDICTION
        # -----------------------
        with torch.no_grad():
            logits, att_weights, _, _ = model(img_tensor, ehr_tensor)

        probs = torch.sigmoid(logits).cpu().numpy()[0]
        preds = (probs > 0.5).astype(int)

        # -----------------------
        # SHOW PREDICTIONS
        # -----------------------
        st.subheader(" Predictions")
        found = False
        for i, val in enumerate(preds):
            if val == 1:
                st.write(f"{inv_label_map[i]} (Prob: {probs[i]:.2f})")
                found = True

        if not found:
            st.write("No disease detected")

        # -----------------------
        # ATTENTION
        # -----------------------
        att = att_weights.squeeze().cpu().numpy()
        st.subheader(" Attention Weights")
        st.write(f"Image Importance: {att[0].mean():.3f}")
        st.write(f"EHR Importance: {att[1].mean():.3f}")

        # -----------------------
        # GRAD-CAM
        # -----------------------
        st.subheader(" Grad-CAM")

        import cv2

        # Generate CAM
        cam = generate_gradcam(model, (img_tensor, ehr_tensor), model.img_enc.features[-1])

       

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Resize
        cam = cv2.resize(cam, (256, 256))

        # Convert original image
        img_np = img.resize((256, 256))
        img_np = np.array(img_np)

        # Convert grayscale if needed
        if len(img_np.shape) == 2:
            img_np = np.stack([img_np]*3, axis=-1)

        # Create heatmap
        heatmap = (cam * 255).astype("uint8")
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

        # Show
        st.image(overlay, caption="Grad-CAM", use_container_width=True)
        # -----------------------
        # FAIRNESS CHECK (RAI)
        # -----------------------
        fairness_score = check_fairness(model, img_tensor, ehr_tensor, DEVICE)

        st.subheader(" Fairness Check")

        if fairness_score < 0.1:
            st.success(f"Model is fair  (Difference: {fairness_score:.3f})")
        else:
            st.warning(f"Potential bias detected  (Difference: {fairness_score:.3f})")

        # -----------------------
        # LOGGING (RAI)
        # -----------------------
        log_prediction("uploaded_image", preds.tolist(), probs.tolist(), att.tolist())

        st.success("Prediction complete & logged ")