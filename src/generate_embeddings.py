import torch
import cv2
import os
import pickle
import numpy as np
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Config
FACES_FOLDER = 'data/faces'
DB_PATH = 'database/embeddings.pkl'

def generate_embeddings():
    # 1. Initialize FaceNet (The good model)
    # We use 'vggface2' weights which are great for identification
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    
    # Standard preprocessing for FaceNet
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    embeddings_dict = {}

    if not os.path.exists('database'):
        os.makedirs('database')

    print(f"Generating embeddings from {FACES_FOLDER}...")
    
    # Get list of face images
    face_files = [f for f in os.listdir(FACES_FOLDER) if f.endswith(('.jpg', '.png'))]
    
    if not face_files:
        print("No faces found! Run detect_faces.py first.")
        return

    for face_file in tqdm(face_files):
        img_path = os.path.join(FACES_FOLDER, face_file)
        
        try:
            img = Image.open(img_path).convert('RGB')
            
            # Prepare image for AI model
            img_tensor = transform(img).unsqueeze(0)

            # Generate embedding
            with torch.no_grad():
                embedding = resnet(img_tensor).detach().numpy()[0]
            
            # Store in simple dictionary format
            embeddings_dict[face_file] = embedding
            
        except Exception as e:
            print(f"Skipping {face_file}: {e}")

    # Save to disk
    with open(DB_PATH, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    
    print(f"✅ Success! Saved {len(embeddings_dict)} embeddings to {DB_PATH}")

if __name__ == "__main__":
    generate_embeddings()