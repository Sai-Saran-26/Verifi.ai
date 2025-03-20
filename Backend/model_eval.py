from infer import predict_video
import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ======================
# Evaluation Function
# ======================
def evaluate_model(real_videos_folder, fake_videos_folder):
    """Evaluates the model on real and fake video datasets."""
    
    y_true = []  # Ground truth labels (1 for real, 0 for fake)
    y_pred = []  # Model predictions
    print(f"Real Videos")
    # Process real videos
    for video in os.listdir(real_videos_folder):
        video_path = os.path.join(real_videos_folder, video)
        # print(video_path)
        try:
            prediction, _ = predict_video(video_path)
        except:
            continue
        y_true.append(1)  # Real = 1
        y_pred.append(0 if prediction == "Real" else 1)
    print("Fake Videos")
    # Process fake videos
    for video in os.listdir(fake_videos_folder):
        video_path = os.path.join(fake_videos_folder, video)
        try:
            prediction, _ = predict_video(video_path)
        except:
            continue
        y_true.append(0)  # Fake = 0
        y_pred.append(1 if prediction == "Real" else 0)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\nEvaluation Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    return accuracy, precision, recall, f1

# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    real_videos_folder = r"C:\Users\rames\myprojects\deepFake\DeepFake_own\test_og"
    fake_videos_folder = r"C:\Users\rames\myprojects\deepFake\DeepFake_own\test_fake"

    if not os.path.exists(real_videos_folder) or not os.path.exists(fake_videos_folder):
        print("Error: One or both video folders do not exist.")
    else:
        evaluate_model(real_videos_folder, fake_videos_folder)

