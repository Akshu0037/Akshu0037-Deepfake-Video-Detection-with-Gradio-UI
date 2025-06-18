# Deepfake Face Classification with ResNeXt101 and PyTorch

# Block 1: Install & import necessary packages
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Block 2: Face-aware frame extraction function
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def extract_faces_from_video(
    video_path,
    frame_count=10,
    output_size=(128,128),
    face_cascade=face_cascade
):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // frame_count, 1)

    faces = []
    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5
        )
        if len(dets) > 0:
            x, y, w, h = max(dets, key=lambda r: r[2]*r[3])
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, output_size)
            faces.append(face)
    cap.release()
    return faces  # list of 10 cropped face images or fewer

# Block 3: Prepare face dataset (extract, label, split)
REAL_VID_DIR = "/kaggle/input/faceforensics/FF++/real"
FAKE_VID_DIR = "/kaggle/input/faceforensics/FF++/fake"
FRAME_COUNT = 10
TMP_DIR = "./faces_tmp"
os.makedirs(TMP_DIR, exist_ok=True)

filepaths, labels = [], []

# extract faces and save images
for label, vid_dir in [(0, REAL_VID_DIR), (1, FAKE_VID_DIR)]:
    for vf in tqdm(os.listdir(vid_dir)[:200], desc=f"Processing {'Real' if label==0 else 'Fake'}"):
        vid_path = os.path.join(vid_dir, vf)
        faces = extract_faces_from_video(vid_path, FRAME_COUNT)
        if len(faces) == FRAME_COUNT:
            for idx, face in enumerate(faces):
                out_path = os.path.join(TMP_DIR, f"{label}_{vf}_{idx}.jpg")
                cv2.imwrite(out_path, face)
                filepaths.append(out_path)
                labels.append(label)

# train/val split
X_train, X_val, y_train, y_val = train_test_split(
    filepaths, labels, test_size=0.2, stratify=labels, random_state=42
)

# Block 4: Dataset and DataLoader
class FaceFrameDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

train_ds = FaceFrameDataset(X_train, y_train, transform)
val_ds   = FaceFrameDataset(X_val,   y_val,   transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)
# Block 5: Load ResNeXt101 classifier
model = models.resnext101_32x8d(pretrained=True)
for param in model.parameters(): param.requires_grad = False
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 2)
)
model = model.to(device)

# Block 6: Training loop
def train(model, train_loader, val_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)
    for epoch in range(epochs):
        model.train()
        total_loss, total_correct = 0, 0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            total_correct += (outputs.argmax(1)==lbls).sum().item()
        print(f"Train Loss: {total_loss/len(train_loader.dataset):.4f}",
              f"Acc: {total_correct/len(train_loader.dataset):.4f}")

        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                val_loss += criterion(out, lbls).item() * imgs.size(0)
                val_correct += (out.argmax(1)==lbls).sum().item()
        print(f"Val Loss: {val_loss/len(val_loader.dataset):.4f}",
              f"Val Acc: {val_correct/len(val_loader.dataset):.4f}\n")

# Block 7: Run training
train(model, train_loader, val_loader, epochs=40)
# Block 8: Save model
torch.save(model.state_dict(), "resnext101_deepfake_faces.pth")
# === Block: Model Testing & Evaluation ===

import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Rebuild dataset & loader for “test” splits:
#    If you already have X_test & y_test lists from your split, skip this and just wrap them,
#    otherwise you can reuse the FaceFrameDataset class and point it at your saved faces.
test_ds = FaceFrameDataset(X_val, y_val, transform)   # or use X_test,y_test if you set aside a test split
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# 2) Load the trained weights
model.load_state_dict(torch.load("resnext101_deepfake_faces.pth", map_location=device))
model.eval()

# 3) Collect predictions and true labels
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to(device)
        out = model(imgs)
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(lbls.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# 4) Compute metrics
acc = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {acc*100:.2f}%\n")

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["REAL","FAKE"]))

# 5) Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", 
            xticklabels=["REAL","FAKE"], 
            yticklabels=["REAL","FAKE"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

import torch

# 1) Ensure your model and utilities are loaded:
#    - model: your ResNeXt101 face‐crop classifier (already .to(device))
#    - extract_faces_from_video(), transform, device from earlier code

# Load trained weights
model.load_state_dict(torch.load("resnext101_deepfake_faces.pth", map_location=device))

# 2) Define the sample video path (replace with your file)
sample_path = "/kaggle/input/faceforensics/FF++/fake/01_02__outside_talking_still_laughing__YVGY8LOK.mp4"

# 3) Run prediction
label, confidence, details = predict_video(
    sample_path,
    model,
    transform,
    device,
    frame_count=10,
    return_details=True
)

# 4) Print results
print("=== Video Deepfake Prediction ===")
print(f"Video File : {sample_path}")
print(f"Prediction : {label}")
print(f"Confidence : {confidence:.2f}")
print("\nPer-frame probabilities:")
for idx, (r_prob, f_prob) in enumerate(zip(details["real_probs"], details["fake_probs"])):
    print(f" Frame {idx+1:2d} → REAL: {r_prob:.2f}, FAKE: {f_prob:.2f}")

import torch

# 1) Ensure your model and utilities are loaded:
#    - model: your ResNeXt101 face‐crop classifier (already .to(device))
#    - extract_faces_from_video(), transform, device from earlier code

# Load trained weights
model.load_state_dict(torch.load("resnext101_deepfake_faces.pth", map_location=device))

# 2) Define the sample video path (replace with your file)
sample_path = "/kaggle/input/faceforensics/FF++/real/01__podium_speech_happy.mp4"

# 3) Run prediction
label, confidence, details = predict_video(
    sample_path,
    model,
    transform,
    device,
    frame_count=10,
    return_details=True
)

# 4) Print results
print("=== Video Deepfake Prediction ===")
print(f"Video File : {sample_path}")
print(f"Prediction : {label}")
print(f"Confidence : {confidence:.2f}")
print("\nPer-frame probabilities:")
for idx, (r_prob, f_prob) in enumerate(zip(details["real_probs"], details["fake_probs"])):
    print(f" Frame {idx+1:2d} → REAL: {r_prob:.2f}, FAKE: {f_prob:.2f}")
