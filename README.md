# 👁️ CNN-Based CCTV Person Re-Identification System

> A Computer Vision pipeline that detects faces in CCTV footage and re-identifies individuals using Deep Learning vector embeddings.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-FaceNet-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

## 📌 Project Overview
This system automates the process of analyzing security footage to find specific individuals. Instead of manual review, the system:
1.  **Extracts frames** from video footage.
2.  **Detects and crops faces** using MTCNN (Multi-task Cascaded Convolutional Networks).
3.  **Generates 512-dimensional embeddings** for each face using a pre-trained Inception-Resnet (FaceNet) model.
4.  **Matches a test subject** (e.g., a selfie) against the database using Euclidean distance metrics.

## 🚀 Key Features
* **Deep Learning Inference:** Uses `facenet-pytorch` for state-of-the-art face recognition accuracy.
* **Vector Similarity Search:** Implements Euclidean distance logic to distinguish between identities with high precision (Threshold < 0.6).
* **Privacy-First Design:** Processes data locally without external API calls.
* **Robust Preprocessing:** Includes "Save-and-Reload" normalization to ensure test images match the compression artifacts of video frames.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Computer Vision:** OpenCV (`cv2`), Pillow (`PIL`)
* **AI/ML Frameworks:** PyTorch, torchvision
* **Model Architecture:** InceptionResnetV1 (Pre-trained on VGGFace2)
* **Data Processing:** NumPy

## 📂 Project Structure
```text
cctv_cnn_matching/
├── data/
│   ├── cctv.mp4           # Input video footage
│   ├── frames/            # Extracted video frames
│   └── faces/             # Cropped face datasets
├── database/
│   └── embeddings.pkl     # Vector database (Pickle file)
├── src/
│   ├── extract_frames.py      # Video processing
│   ├── detect_faces.py        # Face detection logic
│   ├── generate_embeddings.py # CNN feature extraction
│   └── run_matching.py        # Main execution pipeline
├── test.jpg               # Target person to find
├── requirements.txt       # Dependencies
└── README.md              # Documentation