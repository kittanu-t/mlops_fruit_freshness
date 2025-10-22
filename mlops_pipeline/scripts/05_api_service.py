import io, yaml, json, torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms, models
import torch.nn as nn

cfg = yaml.safe_load(open("config.yaml"))
app = FastAPI(title="Fruits Freshness API")

# โหลด mapping + โมเดล Production (ใช้ไฟล์ local ที่ log ไว้ตอนเทรน)
class_to_idx = json.load(open("mlruns/0/<PUT_RUN_ID_HERE>/artifacts/artifacts/class_to_idx.json"))  # แก้เป็น path จริง/หรือย้ายไฟล์มากลาง
idx_to_class = {v:k for k,v in class_to_idx.items()}

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
model.eval()

tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(cfg["img_size"]),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1).squeeze().tolist()
        idx = int(torch.argmax(logits, dim=1))
    return {"label": idx_to_class[idx], "prob": prob[idx]}
