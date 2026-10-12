import torch
from covid_dataloader import CovidXrayDataset
from models.multimodal_model import MultimodalModel
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CovidXrayDataset(
    image_dir="../data/covid_xray/images",
    metadata_path="../data/covid_xray/metadata_clean.csv",
    label_map_path="disease_labels.json"
)

model = MultimodalModel().to(device)
model.load_state_dict(torch.load("saved_models/multimodal_model.pth", map_location=device))
model.eval()

# Pick sample
img, ehr, target = dataset[10]
img_tensor = img.unsqueeze(0).to(device)
ehr_tensor = ehr.unsqueeze(0).to(device)

with torch.no_grad():
    logits, att_weights = model(img_tensor, ehr_tensor)

# att_weights shape is (2, 128)
att = att_weights.squeeze().cpu().numpy()

img_att = att[0]      # Attention for image features (128 dim)
ehr_att = att[1]      # Attention for EHR features (128 dim)

# Plot IMAGE attention
plt.figure(figsize=(10,4))
plt.bar(np.arange(len(img_att)), img_att, color='blue')
plt.title("Attention Weights – Image Features")
plt.xlabel("Feature Index")
plt.ylabel("Attention Weight")
plt.tight_layout()
plt.savefig("attention_image.png")
plt.show()

# Plot EHR attention
plt.figure(figsize=(10,4))
plt.bar(np.arange(len(ehr_att)), ehr_att, color='green')
plt.title("Attention Weights – EHR Features")
plt.xlabel("Feature Index")
plt.ylabel("Attention Weight")
plt.tight_layout()
plt.savefig("attention_ehr.png")
plt.show()

print("Saved attention_image.png and attention_ehr.png")
