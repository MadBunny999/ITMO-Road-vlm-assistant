from pathlib import Path
import sys
import os

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = os.path.relpath(root_path)

data_dir = os.path.join(ROOT, "data")
processed_data_dir = os.path.join(data_dir, "processed")

images_dir = os.path.join(processed_data_dir, "images")
raw_labels_path = os.path.join(processed_data_dir, "labels.csv")
yolo_labels_dir = os.path.join(processed_data_dir, "labels")

# Sources
VIDEO = 'Video'
WEBCAM = 'Webcam'
RTSP = 'RTSP'
YOUTUBE = 'YouTube'

SOURCES_LIST = [VIDEO, WEBCAM, RTSP, YOUTUBE]

# Webcam
WEBCAM_PATH = 0

YOLO_PATH = os.path.join(ROOT, "models/best.pt")

VLM_URL = "http://127.0.0.1:8000"
PROMPT = "Describe driver's actions according to road signs"
