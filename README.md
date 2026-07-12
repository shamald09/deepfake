# Deepfake Detector

Deepfake Detector is a lightweight image forensics project with:
- HTML/CSS/JS frontend for image upload and result display
- Flask backend for file handling and model inference
- PyTorch (`segmentation-models-pytorch`) DeepfakeNet classifier for `REAL`, `DEEPFAKE`, or `AI-GENERATED`

## Project Structure
- `frontend/`: HTML, CSS, and JS for the upload UI
- `backend/`: Flask API and model loading/inference code
- `backend/model/`: trained DeepfakeNet checkpoint

## Current Detection Flow
1. User drags or selects an image in the frontend.
2. The frontend sends the image to the Flask backend's `/predict` endpoint.
3. The backend runs the image through the DeepfakeNet model and returns:
   - `result` — `Real`, `Deepfake`, or `AI-Generated`
   - `confidence` — the model's confidence score (0–1)
   - `fake_type` — the model's fine-grained guess (`real`, `face_swap`, `face_reenactment`, or `full_ai_generated`)
4. The frontend renders the verdict and an animated confidence bar.

## Install

### Frontend
No install step — it's static HTML/CSS/JS with no build tooling.

### Backend
```powershell
cd backend
pip install -r requirements.txt
```

## Run The App

### Start Backend
```powershell
cd backend
python app.py
```
The backend runs on `http://127.0.0.1:5000`.

### Start Frontend
```powershell
cd frontend
python -m http.server 5500
```
Open `http://127.0.0.1:5500/index.html` in your browser.
Opening `index.html` directly by double-clicking can trigger browser CORS restrictions — serving it locally like this avoids that.

## Model

The model is **DeepfakeNet**, a `smp.Unet` subclass (`encoder_name="timm-efficientnet-b4"`) with three output heads: a segmentation head (manipulation mask), a classification head (real/fake), and a type head (`real`, `face_swap`, `face_reenactment`, `full_ai_generated`). It was trained on Kaggle using a merged, class-balanced manifest from four datasets — `xdxd003/ff-c23` (FaceForensics++), `xhlulu/140k-real-and-fake-faces`, `birdy654/cifake-real-and-ai-generated-synthetic-images`, and `itamargr/dfdc-faces-of-the-train-sample` — with per-epoch validation and automatic early stopping on overfitting.

Checkpoint format is a raw PyTorch `state_dict`:
```text
backend/model/
  best_model.pt
```

### Run A Single Prediction (without the UI)
```powershell
python -c "
import torch
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
    n_types=4, encoder_name='timm-efficientnet-b4', encoder_weights=None,
    in_channels=3, classes=1, aux_params=dict(pooling='avg', dropout=0.3, classes=1),
)
model.load_state_dict(torch.load('backend/model/best_model.pt', map_location='cpu', weights_only=True))
model.eval()

tf = A.Compose([A.Resize(320, 320), A.Normalize(), ToTensorV2()])
img = np.array(Image.open('test.jpg').convert('RGB'))
x = tf(image=img)['image'].unsqueeze(0)

with torch.no_grad():
    mask_logits, cls_logit, type_logits = model(x)
    fake_prob = torch.sigmoid(cls_logit).item()
    print('fake probability:', fake_prob)
"
```

## Notes
- The model detects real vs. AI-generated images **and** face-swap/reenactment-style deepfakes — unlike a plain real-vs-AI classifier, it does not require a separately trained model for that category.
- The confidence score reflects the model's own certainty, not a guarantee of correctness — always treat borderline scores (close to 50%) as inconclusive.
- The segmentation mask is a model-predicted region estimate, not a ground-truth annotation — treat it as an explanation aid, not a certainty.
- No internet access is required at runtime since `encoder_weights=None` skips downloading ImageNet initialization — only the local `best_model.pt` weights are loaded.
