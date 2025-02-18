import os
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, flash

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = set(['mp4', 'avi', 'mov'])
app.secret_key = 'your-secret-key'

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

def allowed_file(filename):
    """Check if the uploaded file has an allowed video extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_frame(pil_image):
    """Predict the label for a single PIL image (frame)."""
    image = pil_image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
    # Assuming class index 1 corresponds to "Indian Police"
    return "Indian Police" if pred.item() == 1 else "Not Indian Police"

def process_video(video_path, frame_interval=30):
    """
    Process the video file frame-by-frame.
    Samples one frame every `frame_interval` frames.
    Returns a result message along with frame statistics.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    police_count = 0
    total_frames_processed = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            # Convert frame (BGR from OpenCV) to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            label = predict_frame(pil_image)
            if label == "Indian Police":
                police_count += 1
            total_frames_processed += 1
        frame_count += 1
    
    cap.release()
    
    # If any frame shows an Indian Police, flag as scam.
    result = "Could be a scam" if police_count > 0 else "Not a scam"
    return result, police_count, total_frames_processed

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename  # For production, consider using secure_filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)
            result, police_count, total_frames = process_video(filepath)
            return render_template('result.html', result=result, police_count=police_count, total_frames=total_frames)
        else:
            flash('File type not allowed')
            return redirect(request.url)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
