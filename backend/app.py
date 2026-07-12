"""
Flask backend for the AI vs Real Detector UI — powered by DeepfakeNet
(smp.Unet subclass, EfficientNet-B4 encoder + fake-type head): best_model.pt
"""

import os
import io
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 320
TYPE_NAMES = ["real", "face_swap", "face_reenactment", "full_ai_generated"]

# ---------------------------------------------------------------------------
# Model — subclasses smp.Unet directly, so encoder/decoder/segmentation_head/
# classification_head are inherited top-level attributes (no "backbone." prefix),
# matching the checkpoint's actual keys exactly. type_head is added on top.
# ---------------------------------------------------------------------------
class DeepfakeNet(smp.Unet):
    def __init__(self, n_types=4, **kwargs):
        super().__init__(**kwargs)
        enc_channels = self.encoder.out_channels[-1]
        self.type_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(enc_channels, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, n_types)
        )

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(features)      # <-- no more "*" unpacking
        mask_logits = self.segmentation_head(decoder_output)
        cls_logit = self.classification_head(features[-1])
        type_logits = self.type_head(features[-1])
        return mask_logits, cls_logit, type_logits


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "best_model.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")
model = DeepfakeNet(
    n_types=4,
    encoder_name="timm-efficientnet-b4",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    aux_params=dict(pooling="avg", dropout=0.3, classes=1),
).to(device)

state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded.")

val_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2(),
])

app = Flask(__name__)
CORS(app)


@torch.no_grad()
def run_inference(pil_image: Image.Image, threshold=0.5):
    img = np.array(pil_image.convert("RGB"))
    inp = val_tf(image=img)["image"].unsqueeze(0).to(device)

    mask_logits, cls_logit, type_logits = model(inp)
    fake_prob = torch.sigmoid(cls_logit).item()
    type_probs = torch.softmax(type_logits, dim=1).squeeze().cpu().numpy()
    pred_type = TYPE_NAMES[int(type_probs.argmax())]

    if fake_prob > threshold:
        result = "AI-Generated" if pred_type == "full_ai_generated" else "Deepfake"
        confidence = fake_prob
    else:
        result = "Real"
        confidence = 1 - fake_prob

    return result, confidence, pred_type


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        image = Image.open(io.BytesIO(file.read()))
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    try:
        result, confidence, fake_type = run_inference(image)
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Could not process image"}), 400

    return jsonify({
        "result": result,
        "confidence": confidence,
        "fake_type": fake_type
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)