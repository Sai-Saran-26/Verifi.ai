import os
import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Define paths
celeb_real_path = r'C:\Users\rames\myprojects\deepFake\DeepFake_own\dataset\Celeb-real'
celeb_synthesis_path = r'C:\Users\rames\myprojects\deepFake\DeepFake_own\dataset\Celeb-synthesis'
youtube_real_path = r'C:\Users\rames\myprojects\deepFake\DeepFake_own\dataset\YouTube-real'
output_frames_path = r'C:\Users\rames\myprojects\deepFake\DeepFake_own\dataset_frames'

# Create output directories
os.makedirs(os.path.join(output_frames_path, 'train/real'), exist_ok=True)
os.makedirs(os.path.join(output_frames_path, 'train/fake'), exist_ok=True)
os.makedirs(os.path.join(output_frames_path, 'val/real'), exist_ok=True)
os.makedirs(os.path.join(output_frames_path, 'val/fake'), exist_ok=True)
os.makedirs(os.path.join(output_frames_path, 'test/real'), exist_ok=True)
os.makedirs(os.path.join(output_frames_path, 'test/fake'), exist_ok=True)

# Parameters
IMG_SIZE = (224, 224)  # Resize frames to 224x224
FRAME_RATE = 1  # Extract 1 frame per second
MAX_FRAMES_PER_VIDEO = 30  # Max frames per video

# Initialize CUDA device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Function to extract frames using OpenCV and PyTorch (GPU-accelerated)
def extract_frames(video_path, output_folder, label, video_id):
    try:
        # Read video with OpenCV
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame indices to extract (1 frame per second)
        frame_indices = [int(fps * sec) for sec in range(0, int(total_frames/fps))]
        frame_indices = [idx for idx in frame_indices if idx < total_frames][:MAX_FRAMES_PER_VIDEO]

        # Extract frames
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert frame to PyTorch tensor and move to GPU
                frame = cv2.resize(frame, IMG_SIZE)
                frame = frame / 255.0  # Normalize to [0, 1]
                frame_tensor = torch.tensor(frame).permute(2, 0, 1).float().unsqueeze(0).to(device)

                frames.append(frame_tensor)

        cap.release()

        # Save frames as numpy arrays
        for i, frame_tensor in enumerate(frames):
            frame_filename = os.path.join(output_folder, f'{label}_video{video_id}_frame{i}.npy')
            np.save(frame_filename, frame_tensor.cpu().numpy())  # Move back to CPU for saving

    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")

def process_videos(video_folder, output_folder, label):
    video_paths = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.mp4')]
    for i, video_path in enumerate(video_paths):
        extract_frames(video_path, output_folder, label, i)

# Process real and fake videos
process_videos(celeb_real_path, os.path.join(output_frames_path, 'train/real'), 'real')
process_videos(youtube_real_path, os.path.join(output_frames_path, 'train/real'), 'real')
process_videos(celeb_synthesis_path, os.path.join(output_frames_path, 'train/fake'), 'fake')

# Split data into train, val, and test sets
def split_data(input_folder, output_folder, split_ratio=(0.7, 0.15, 0.15)):
    real_frames = [os.path.join(input_folder, 'real', f) for f in os.listdir(os.path.join(input_folder, 'real'))]
    fake_frames = [os.path.join(input_folder, 'fake', f) for f in os.listdir(os.path.join(input_folder, 'fake'))]

    # Split real frames
    real_train, real_temp = train_test_split(real_frames, train_size=split_ratio[0], random_state=42)
    real_val, real_test = train_test_split(real_temp, test_size=split_ratio[2]/(split_ratio[1]+split_ratio[2]), random_state=42)

    # Split fake frames
    fake_train, fake_temp = train_test_split(fake_frames, train_size=split_ratio[0], random_state=42)
    fake_val, fake_test = train_test_split(fake_temp, test_size=split_ratio[2]/(split_ratio[1]+split_ratio[2]), random_state=42)

    # Move frames to respective folders
    def move_frames(frames, dest_folder):
        for frame in frames:
            os.rename(frame, os.path.join(dest_folder, os.path.basename(frame)))

    move_frames(real_train, os.path.join(output_folder, 'train/real'))
    move_frames(fake_train, os.path.join(output_folder, 'train/fake'))
    move_frames(real_val, os.path.join(output_folder, 'val/real'))
    move_frames(fake_val, os.path.join(output_folder, 'val/fake'))
    move_frames(real_test, os.path.join(output_folder, 'test/real'))
    move_frames(fake_test, os.path.join(output_folder, 'test/fake'))

# Split the data
split_data(os.path.join(output_frames_path, 'train'), output_frames_path)

print("Video preprocessing complete with CUDA acceleration!")
