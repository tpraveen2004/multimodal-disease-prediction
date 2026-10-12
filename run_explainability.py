import torch
from models.multimodal_model import MultimodalModel
from covid_dataloader import CovidXrayDataset
from explainability.gradcam import generate_gradcam
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
dataset = CovidXrayDataset(
    image_dir="../data/covid_xray/images",
    metadata_path="../data/covid_xray/metadata_clean.csv",
    label_map_path="disease_labels.json"
)

# Load model
model = MultimodalModel().to(device)
model.load_state_dict(torch.load("saved_models/multimodal_model.pth", map_location=device))
model.eval()

# Pick one sample for Grad-CAM
img, ehr, target = dataset[10]
img_tensor = img.unsqueeze(0).to(device)
ehr_tensor = ehr.unsqueeze(0).to(device)

# Grad-CAM requires both inputs
print("Generating Grad-CAM...")
cam = generate_gradcam(model, (img_tensor, ehr_tensor), model.img_enc.features[-1])

# Convert image for display
img_np = img.permute(1,2,0).cpu().numpy()
img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

plt.imshow(img_np)
plt.imshow(cam, cmap="jet", alpha=0.5)
plt.title("Grad-CAM Heatmap")
plt.axis("off")
plt.savefig("gradcam_output.png")
plt.show()

print("Grad-CAM saved as gradcam_output.png")
