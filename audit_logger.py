import json
from datetime import datetime

def log_prediction(image_path, preds, probs, attention):
    log = {
        "timestamp": str(datetime.now()),
        "image": image_path,
        "predictions": preds,
        "probabilities": probs,
        "attention": attention
    }

    with open("audit_logs.json", "a") as f:
        f.write(json.dumps(log) + "\n")