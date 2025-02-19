import os
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI()

# Set up template directory
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Setup device and load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'best_model.pth'

# Initialize model architecture (ResNet50) and load weights
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # Assuming two classes: 0 = not_police, 1 = police
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_frame(pil_image: Image.Image) -> str:
    image = pil_image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
    return "Indian Police" if pred.item() == 1 else "Not Indian Police"

def process_video(video_path: str, frame_interval: int = 30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    police_count = 0
    total_frames_processed = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            label = predict_frame(pil_image)
            if label == "Indian Police":
                police_count += 1
            total_frames_processed += 1
        frame_count += 1
    
    cap.release()
    result = "Could be a scam" if police_count > (total_frames_processed / 2) else "Not a scam"
    return result, police_count, total_frames_processed

@app.get("/", response_class=HTMLResponse)
async def get_upload(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload/", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        return templates.TemplateResponse("upload.html", {"request": request, "error": "File type not allowed"})
    
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    result, police_count, total_frames = process_video(file_path)
    
    # Optionally remove the file after processing
    os.remove(file_path)
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "result": result,
        "police_count": police_count,
        "total_frames_processed": total_frames
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
