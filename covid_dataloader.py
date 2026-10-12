import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
import numpy as np
import json

class CovidXrayDataset(Dataset):
    def __init__(self, image_dir, metadata_path, label_map_path):
        self.image_dir = image_dir

        # Load metadata
        self.df = pd.read_csv(metadata_path)

        # Replace NaN with empty strings
        self.df = self.df.replace({np.nan: ""})

        # Load disease label map
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)

        # Detect filename column
        if "filename" in self.df.columns:
            self.image_column = "filename"
        elif "image" in self.df.columns:
            self.image_column = "image"
        else:
            raise ValueError("CSV must contain a 'filename' or 'image' column")

        # Transform for image input
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Allowed image formats
        self.valid_ext = (".png", ".jpg", ".jpeg")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ----------------------------
        # 1. Load Image Safely
        # ----------------------------
        filename = str(row[self.image_column]).strip()
        img_path = os.path.join(self.image_dir, filename)

        # Skip if file is NOT an image
        if not filename.lower().endswith(self.valid_ext):
            return self.__getitem__((idx + 1) % len(self.df))

        # Skip missing image files
        if not os.path.exists(img_path):
            return self.__getitem__((idx + 1) % len(self.df))

        # Try to load image
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            return self.__getitem__((idx + 1) % len(self.df))

        img = self.transform(img)

        # ----------------------------
        # 2. Build EHR Tensor Safely
        # ----------------------------
        sex_value = str(row.get("sex", "")).lower()
        survival_value = str(row.get("survival", "")).lower()
        intubated_value = str(row.get("intubated", "")).lower()

        # Age (handle float/NaN)
        age_raw = row.get("age", 0)
        try:
            age_value = int(float(age_raw))
        except:
            age_value = 0

        ehr_tensor = torch.tensor([
            1 if sex_value == "male" else 0,
            age_value,
            1 if survival_value == "dead" else 0,
            1 if intubated_value == "yes" else 0
        ], dtype=torch.float32)

        # ----------------------------
        # 3. Multi-label Disease Target
        # ----------------------------
        diseases_raw = str(row.get("finding", "")).lower()
        diseases = [d.strip() for d in diseases_raw.split("|")]

        target = torch.zeros(len(self.label_map), dtype=torch.float32)

        for d in diseases:
            if d in self.label_map:
                target[self.label_map[d]] = 1.0

        return img, ehr_tensor, target
