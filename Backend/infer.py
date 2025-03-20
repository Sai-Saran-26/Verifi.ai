import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn

MODEL_PATH = r'C:\Users\rames\myprojects\deepFake\DeepFake_own\src\best_resnet_model.pth'
FRAME_COUNT = 16
IMG_SIZE = (224, 224)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DeepfakeResNet(nn.Module):
    def __init__(self):
        super(DeepfakeResNet, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        batch_size, frame_count, channels, height, width = x.shape
        x = x.view(batch_size * frame_count, channels, height, width)
        x = self.resnet(x)
        x = x.view(batch_size, frame_count)
        x = torch.mean(x, dim=1, keepdim=True)
        return x

model = DeepfakeResNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def extract_frames(video_path, max_frames=FRAME_COUNT):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None

    frame_idxs = np.linspace(0, total_frames - 1, max_frames, dtype=int) if total_frames >= max_frames else range(total_frames)

    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, IMG_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    while len(frames) < max_frames:
        frames.append(frames[-1])

    return np.array(frames)

def preprocess_frames(frames):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    frames = torch.stack([transform(frame) for frame in frames])
    frames = frames.unsqueeze(0)
    return frames

def predict_video(video_path):
    frames = extract_frames(video_path)
    if frames is None:
        print("Error: Could not extract frames from the video.")
        return
    
    frames = preprocess_frames(frames)
    frames = frames.to(DEVICE)

    with torch.no_grad():
        output = model(frames)
        prob = torch.sigmoid(output).item()
        prediction = "Real" if prob > 0.5 else "Fake"

    print(f"Prediction: {prediction} (Confidence: {prob:.4f})")
    return prediction, prob

if __name__ == "__main__":
    video_path = r'C:\Users\rames\myprojects\deepFake\DeepFake_own\test_fake\test7.mp4'
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
    else:
        prediction, confidence = predict_video(video_path)
        print(f"Video is predicted to be: {prediction}")
