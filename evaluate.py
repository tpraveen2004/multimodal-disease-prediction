import torch
from torch.utils.data import DataLoader
from covid_dataloader import CovidXrayDataset
from models.multimodal_model import MultimodalModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

def evaluate_model():
    print("\nLoading dataset...")

    dataset = CovidXrayDataset(
        image_dir="../data/covid_xray/images",
        metadata_path="../data/covid_xray/metadata_clean.csv",
        label_map_path="disease_labels.json"
    )

    print("Dataset size:", len(dataset))

    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    print("Loading model...")
    model = MultimodalModel().to(device)
    model.load_state_dict(torch.load("saved_models/multimodal_model.pth", map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    print("\nRunning evaluation...")

    with torch.no_grad():
        for images, ehr, labels in loader:
            images, ehr = images.to(device), ehr.to(device)
            logits, _ = model(images, ehr)

            preds = (torch.sigmoid(logits) > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Convert to arrays
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print("\n--- FINAL EVALUATION RESULTS ---")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1-score :", f1)
    print("\nEvaluation Complete.")

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    evaluate_model()
