 Deepfake & AI-Generated Image Detector

 It is a full-stack image forensics project with:


HTML/CSS/JS frontend for image upload and result display
Flask backend for file handling and model inference
PyTorch (segmentation-models-pytorch) DeepfakeNet — a Unet with an EfficientNet-B4 encoder, extended with a classification head (REAL/FAKE) and a fake-type head (face-swap, face-reenactment, or fully AI-generated)


Project Structure


frontend/: HTML, CSS, and JS for the upload UI
backend/: Flask API and model loading/inference code
backend/model/: trained checkpoint (best_model.pt)


Current Detection Flow


User drags or selects an image in the frontend.
The frontend sends the image to the Flask backend's /predict endpoint.
The backend runs the image through DeepfakeNet and returns:

result — Real, Deepfake, or AI-Generated
confidence — the model's confidence score (0–1)
fake_type — the model's fine-grained guess (real, face_swap, face_reenactment, or full_ai_generated)



The frontend renders the verdict and an animated confidence bar.


Install

Frontend

No install step — static HTML/CSS/JS, no build tooling.

Backend

powershellcd backend
pip install -r requirements.txt

Run The App

Start Backend

powershellcd backend
python app.py

The backend runs on http://127.0.0.1:5000.

Start Frontend

powershellcd frontend
python -m http.server 5500

Open http://127.0.0.1:5500/index.html in your browser.
Opening index.html directly by double-clicking can trigger browser CORS restrictions — serving it locally like this avoids that.

Model

DeepfakeNet is a smp.Unet subclass (encoder_name="timm-efficientnet-b4") with three output heads:


Segmentation head — pixel-level manipulation mask (inherited from smp.Unet)
Classification head — binary real/fake logit (inherited aux_params head)
Type head — 4-way classifier: real, face_swap, face_reenactment, full_ai_generated


It was trained on a merged, class-balanced manifest built from four Kaggle datasets:


FaceForensics++ (C23) — face-swap and face-reenactment video frames
140k Real and Fake Faces — GAN face-swap style images
CIFAKE — diffusion-generated vs. real images
DFDC Faces — real-world video face swaps


Training used per-epoch validation with automatic early stopping: it tracks val loss, and if val loss doesn't improve for PATIENCE epochs in a row, training stops and the checkpoint with the best val loss (not the last epoch) is kept as best_model.pt — this is the file the backend loads.

Checkpoint format is a raw PyTorch state_dict:

textbackend/model/
  best_model.pt

Run A Single Prediction (without the UI)

pythonimport torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn

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
        decoder_output = self.decoder(features)
        mask_logits = self.segmentation_head(decoder_output)
        cls_logit = self.classification_head(features[-1])
        type_logits = self.type_head(features[-1])
        return mask_logits, cls_logit, type_logits

model = DeepfakeNet(
    n_types=4, encoder_name="timm-efficientnet-b4", encoder_weights=None,
    in_channels=3, classes=1, aux_params=dict(pooling="avg", dropout=0.3, classes=1),
)
model.load_state_dict(torch.load("backend/model/best_model.pt", map_location="cpu", weights_only=True))
model.eval()

tf = A.Compose([A.Resize(320, 320), A.Normalize(), ToTensorV2()])
img = np.array(Image.open("test.jpg").convert("RGB"))
x = tf(image=img)["image"].unsqueeze(0)

with torch.no_grad():
    mask_logits, cls_logit, type_logits = model(x)
    print("fake prob:", torch.sigmoid(cls_logit).item())

Notes
The highlighted/segmentation mask is a model-predicted region estimate, not a ground-truth annotation — treat it as an explanation aid, not a certainty.
Confidence near 50% should be treated as inconclusive.
FaceForensics++ frames in training use whole-face-crop masks (mask_path="FULL") where the specific upload didn't include ground-truth manipulation masks — so localization is coarser for those samples than for ones with true pixel-level masks.
The model generalizes best to the categories it saw real training data for (face-swap, face-reenactment, general diffusion images); Ghibli-style filters and body-specific manipulation are weaker unless custom data was added for those categories during training (see Cell 2.5 of the training notebook).
