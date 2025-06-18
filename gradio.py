# Import necessary libraries
import gradio as gr
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import os
from PIL import Image
# 1. Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 2. Face Detection Cascade

 This initializes a pre-trained Haar Cascade classifier from OpenCV. This specific classifier is designed to detect frontal faces in images.
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# 3. Face Extraction Function

This function, extract_faces_from_video, takes a video file path and extracts faces from a specified number of frames.
def extract_faces_from_video(
    video_path,
    frame_count=10,
    output_size=(128, 128),
    face_cascade=face_cascade
):
    """
    Extracts faces from a video at regular intervals.
    Returns a list of cropped face images.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Warning: Video file {video_path} has 0 frames.")
        cap.release()
        return []

    step = max(total_frames // frame_count, 1)

    faces = []
    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        dets = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30) # Added minSize for robustness
        )

        if len(dets) > 0:
            # Get the largest face
            x, y, w, h = max(dets, key=lambda r: r[2] * r[3])
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, output_size)
            faces.append(face)

    cap.release()
    return faces
# 4. Image Transformation

This defines a sequence of image transformations that will be applied to the extracted face images before they are fed into the deep learning model.
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# 5. Model Definition and Loading

This section defines the deep learning model architecture and loads pre-trained weights.
def load_model(model_path="/content/deepfake.pth", device=device):
    """
    Loads the ResNeXt101 model with custom final layer and trained weights.
    """
    # Load pre-trained ResNeXt101
    model = models.resnext101_32x8d(pretrained=True)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2) # 2 classes: REAL, FAKE
    )

    # Load saved state_dict
    try:
        # Use map_location to ensure it loads correctly regardless of original device
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from {model_path} to {device}.")
    except FileNotFoundError:
        print(f"Error: Model weights file not found at {model_path}. Please ensure it's in the same directory.")
        # Optionally, handle by training or downloading a default model
        return None # Or raise an error
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return None

    model = model.to(device)
    model.eval() # Set model to evaluation mode
    return model

# Load the model globally to avoid reloading on each prediction
model = load_model()
# 6. Prediction Function for Gradio

This function, predict_deepfake, takes the video file path as input and returns a string indicating whether the video is classified as REAL or FAKE and the confidence level.
def predict_deepfake(video_path):
    """
    Predicts if a video contains a deepfake.
    Returns prediction text for UI, and raw data for state.
    """
    if model is None:
        return "Error: Model not loaded. Cannot perform prediction.", None, None, None

    try:
        faces = extract_faces_from_video(video_path)

        if not faces:
            print("Debug: No faces detected in the video.")
            return "No faces detected in the video. Cannot classify.", None, None, None

        face_tensors = [transform(Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))) for face in faces]

        if not face_tensors:
            print("Debug: Error during face tensor creation.")
            return "Error during face processing.", None, None, None

        inputs = torch.stack(face_tensors).to(device)

        with torch.no_grad():
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            avg_probabilities = torch.mean(probabilities, dim=0)
            predicted_class_idx = torch.argmax(avg_probabilities).item()
            confidence = avg_probabilities[predicted_class_idx].item()

        labels = ["REAL", "FAKE"]
        predicted_label_str = labels[predicted_class_idx]

        result_text = f"Prediction: {predicted_label_str}\nConfidence: {confidence:.2f}"

        # Return prediction details for the state component
        return result_text, video_path, predicted_label_str, confidence

    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        return f"Prediction Error: An unexpected error occurred. Details: {e}", None, None, None
# 7. CSV File Setup
 It defines the filename and headers for a CSV file that stores flagged videos information
# --- CSV File Setup ---
FLAGGED_VIDEOS_CSV = "flagged_videos.csv"
CSV_HEADERS = ["timestamp", "video_path", "model_prediction", "model_confidence", "user_flag"]

# Initialize CSV file with headers if it doesn't exist
if not os.path.exists(FLAGGED_VIDEOS_CSV):
    df = pd.DataFrame(columns=CSV_HEADERS)
    df.to_csv(FLAGGED_VIDEOS_CSV, index=False)
# 8. Flag Function
The function takes the video path, model prediction, model confidence, and user's true label as input. It records this information along with a timestamp and appends it to the csv file we defined earlier.
# --- Flagging Function ---
def flag_video(
    video_path,
    model_prediction,
    model_confidence,
    user_true_label
):
    """
    Flags the video information to a CSV file if there's a mismatch or it's flagged by user.
    """
    if not video_path: # No video uploaded yet
        return "Please upload and get a prediction first."
    if user_true_label == "Not specified": # User didn't select
        return "Please select 'This is REAL' or 'This is FAKE' to flag."

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create a DataFrame for the new entry
    new_entry = pd.DataFrame([{
        "timestamp": timestamp,
        "video_path": video_path,
        "model_prediction": model_prediction,
        "model_confidence": f"{model_confidence:.2f}", # Format confidence
        "user_flag": user_true_label
    }])

    # Append to CSV
    try:
        if not os.path.exists(FLAGGED_VIDEOS_CSV):
            new_entry.to_csv(FLAGGED_VIDEOS_CSV, index=False)
        else:
            new_entry.to_csv(FLAGGED_VIDEOS_CSV, mode='a', header=False, index=False)

        # Check if it's a mismatch for the feedback message
        if model_prediction != user_true_label:
            return f"Video flagged as a MISMATCH!\nDetails saved to {FLAGGED_VIDEOS_CSV}"
        else:
            return f"Video flagged (model prediction matches user's view).\nDetails saved to {FLAGGED_VIDEOS_CSV}"

    except Exception as e: # Catch any exception during CSV operations
        print(f"Error during flagging to CSV: {e}") # Print to console for debugging
        return f"Error saving flag to CSV: {e}"
# 9. Gradio Interface
# --- 7. Gradio Interface ---
with gr.Blocks() as iface:
  #markdown for title and description
    gr.Markdown(
        """
        # Deepfake Video Detector
        Upload a video to detect if it contains a deepfake using a ResNeXt101 model.
        You can also flag videos if you believe the model's prediction is incorrect.
        Ensure 'resnext101_deepfake_faces.pth' and 'haarcascade_frontalface_default.xml' are available.
        """
    )

    with gr.Row():
        video_input = gr.Video(label="Upload Video")
        result_output = gr.Textbox(label="Deepfake Detection Result")

    # Hidden state components to pass information between prediction and flagging
    video_path_state = gr.State(value=None)
    model_prediction_state = gr.State(value=None)
    model_confidence_state = gr.State(value=None)

    # Clear states and result when a new video input is initiated
    video_input.change(
        fn=lambda x: [None, None, None, None, "Ready for prediction..."],
        inputs=video_input, # Use video_input as input to capture change event
        outputs=[video_path_state, model_prediction_state, model_confidence_state, result_output],
        queue=False # This listener should run quickly
    )

    # When video input changes, trigger prediction and update state variables
    video_input.upload(
        fn=predict_deepfake,
        inputs=video_input,
        outputs=[result_output, video_path_state, model_prediction_state, model_confidence_state],
        show_progress=True
    )
    # Also handle the case where video is uploaded via drag-and-drop or Browse (not just changing content)
    # This 'change' listener will trigger predict_deepfake if the video content itself changes.
    # It might be redundant with 'upload' if 'upload' covers all user interactions for input.
    # We will keep it for robustness, but ensure outputs are cleared on *any* change.
    video_input.change(
        fn=predict_deepfake,
        inputs=video_input,
        outputs=[result_output, video_path_state, model_prediction_state, model_confidence_state],
        show_progress=True
    )


    gr.Markdown(
        """
        ### Flag Inaccurate Predictions
        If you believe the model's prediction is incorrect, you can flag the video here.
        This helps in potential future model improvements.
        """
    )
    with gr.Row():
        user_label_radio = gr.Radio(
            ["This is REAL", "This is FAKE"],
            label="What is the TRUE label of this video?",
            value="Not specified" # Default state
        )
        flag_button = gr.Button("Flag Video")
        flag_status_output = gr.Textbox(label="Flag Status")

    # Flag button click event
    flag_button.click(
        fn=flag_video,
        inputs=[video_path_state, model_prediction_state, model_confidence_state, user_label_radio],
        outputs=flag_status_output
    )

    # Clear user radio selection after flagging (optional, but good UX)
    flag_button.click(
        fn=lambda: "Not specified",
        inputs=None,
        outputs=user_label_radio,
        queue=False # Do not wait for this to finish
    )


# Launch the interface
if __name__ == "__main__":
    print(f"Running on device: {device}")
    iface.launch()
