# 👁️ VisionGuard-Re-ID
### CNN-Based CCTV Person Re-Identification System

> A Computer Vision pipeline that detects faces in CCTV footage and re-identifies individuals using Deep Learning vector embeddings.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-FaceNet-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

---

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

---

## 📋 Prerequisites

> [!IMPORTANT]
> This system relies on heavy computer vision operations. While it is fully functional on standard CPUs, a dedicated GPU (**Nvidia CUDA** or **Apple Silicon MPS**) is highly recommended for processing long video footage.

| Tool | Version | Description | Check Installation |
| :--- | :--- | :--- | :--- |
| **Python** | 3.10+ | Core runtime environment | `python --version` |
| **pip** | Latest | Python package manager | `pip --version` |

---

## ⚙️ Setup & Installation

### 1. Configure Environment Variables
Since this system features a privacy-first design with local inference, no external API keys are required. Use the `.env` file to manage your local file paths and sensitivity thresholds.

```bash
# Copy the example configuration file
cp .env.example .env


2. Install Dependencies
It is highly recommended to use a virtual environment to manage dependencies.

Bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
🏃 Usage Guide
The system runs in a sequential pipeline. Execute the scripts in the following order:

Extract Frames: Convert video footage into image sequences.
python src/extract_frames.py

Detect Faces: Isolate and crop faces from the extracted frames.
python src/detect_faces.py

Generate Embeddings: Convert faces into 512-d mathematical vectors.
python src/generate_embeddings.py

Run Matching: Compare your target subject against the processed database.
python src/run_matching.py

📂 Project Structure
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
