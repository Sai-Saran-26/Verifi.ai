import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from loguru import logger

VIDEO_PATHS = {
    'real': [
        r'C:\Users\rames\myprojects\deepFake\DeepFake_own\dataset\Celeb-real',
        r'C:\Users\rames\myprojects\deepFake\DeepFake_own\dataset\YouTube-real'
    ],
    'fake': [
        r'C:\Users\rames\myprojects\deepFake\DeepFake_own\dataset\Celeb-synthesis'
    ]
}

FRAME_COUNT = 16
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

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

def prepare_dataset():
    all_frames = []
    labels = []
    
    for path in VIDEO_PATHS['real']:
        for video_file in os.listdir(path):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(path, video_file)
                frames = extract_frames(video_path)
                if frames is not None:
                    all_frames.append(frames)
                    labels.append(0)

    for path in VIDEO_PATHS['fake']:
        for video_file in os.listdir(path):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(path, video_file)
                frames = extract_frames(video_path)
                if frames is not None:
                    all_frames.append(frames)
                    labels.append(1)

    all_frames = np.array(all_frames)
    labels = np.array(labels)

    train_frames, test_frames, train_labels, test_labels = train_test_split(
        all_frames, labels, test_size=0.2, stratify=labels, random_state=42
    )
    train_frames, val_frames, train_labels, val_labels = train_test_split(
        train_frames, train_labels, test_size=0.1, stratify=train_labels, random_state=42
    )

    return (train_frames, train_labels), (val_frames, val_labels), (test_frames, test_labels)

class VideoFrameDataset(Dataset):
    def __init__(self, frames, labels, transform=None):
        self.frames = frames
        self.labels = labels
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        video_frames = self.frames[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
        transformed_frames = torch.stack([self.transform(frame) for frame in video_frames])
        return transformed_frames, label

class DeepfakeResNet(nn.Module):
    def __init__(self):
        super(DeepfakeResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        batch_size, frame_count, channels, height, width = x.shape
        x = x.view(batch_size * frame_count, channels, height, width)
        x = self.resnet(x)
        x = x.view(batch_size, frame_count)
        x = torch.mean(x, dim=1, keepdim=True)
        return x

def train(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS):
    model.to(DEVICE)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for frames, labels in train_loader:
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(DEVICE), labels.to(DEVICE)
                outputs = model(frames)
                val_loss += criterion(outputs, labels).item()

        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_resnet_model_1.pth')

def main():
    (train_frames, train_labels), (val_frames, val_labels), _ = prepare_dataset()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    num_workers = 0 if os.name == 'nt' else 4

    train_dataset = VideoFrameDataset(train_frames, train_labels, transform)
    val_dataset = VideoFrameDataset(val_frames, val_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    model = DeepfakeResNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, train_loader, val_loader, criterion, optimizer)

if __name__ == "__main__":
    main()
