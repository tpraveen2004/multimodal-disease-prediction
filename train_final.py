import torch
from torch.utils.data import DataLoader
from models.multimodal_model import MultimodalModel
from covid_dataloader import CovidXrayDataset
from training.utils import train_one_epoch
import os

# -----------------------
# DEVICE
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------
# TRAIN FUNCTION
# -----------------------
def train():

    # -----------------------
    # LOAD DATASET
    # -----------------------
    dataset = CovidXrayDataset(
        image_dir="../data/covid_xray/images",
        metadata_path="../data/covid_xray/metadata_clean.csv",
        label_map_path="disease_labels.json"
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # -----------------------
    # MODEL
    # -----------------------
    model = MultimodalModel().to(device)

    # -----------------------
    # OPTIMIZER
    # -----------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # -----------------------
    # TRAINING
    # -----------------------
    EPOCHS = 5

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, loader, optimizer, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}")

    # -----------------------
    # SAVE MODEL (RAI: VERSIONING)
    # -----------------------
    os.makedirs("saved_models", exist_ok=True)

    torch.save({
        "model_state": model.state_dict(),
        "version": "v1.0",
    }, "saved_models/multimodal_model.pth")

    print("Model saved successfully.")

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    train()