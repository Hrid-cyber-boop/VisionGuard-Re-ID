import torch
import cv2
import os
import pickle
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision import transforms

# Setup
TEST_IMG_PATH = 'test.jpg'  # Make sure this matches your file name!
DB_PATH = 'database/embeddings.pkl'

def main():
    print(f"--- DEBUGGING {TEST_IMG_PATH} ---")
    
    # 1. Load Tools
    mtcnn = MTCNN(keep_all=False, select_largest=True, device='cpu')
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    
    # Same transform as used in the database generation
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 2. Check Image
    try:
        img = Image.open(TEST_IMG_PATH).convert('RGB')
        print("Image loaded successfully.")
    except FileNotFoundError:
        print(f"❌ ERROR: Could not find {TEST_IMG_PATH}")
        return

    # 3. Detect Face and SAVE IT to verify
    # We save the crop first to see what the AI is actually looking at
    img_cropped_path = 'debug_test_crop.jpg'
    face_crop = mtcnn(img, save_path=img_cropped_path)

    if face_crop is None:
        print("❌ NO FACE DETECTED! The AI cannot find a face in test.jpg.")
        print("Try a clearer photo with good lighting.")
        return
    else:
        print(f"✅ Face detected! Crop saved to '{img_cropped_path}'")
        print("👉 GO CHECK THAT FILE NOW. Is it your face? Is it centered?")

    # 4. Generate Embedding (Strictly matching DB process)
    # We load the crop we just saved to ensure identical processing
    crop_loader = Image.open(img_cropped_path).convert('RGB')
    img_tensor = transform(crop_loader).unsqueeze(0)

    with torch.no_grad():
        test_embedding = resnet(img_tensor).detach().numpy()[0]

    # 5. Compare with Database
    with open(DB_PATH, 'rb') as f:
        db = pickle.load(f)
    
    # Handle different database structures (simple dict vs nested dict)
    if 'embeddings' in db:
        database_embeddings = {k: v['embedding'] for k, v in db['embeddings'].items()}
    else:
        database_embeddings = db

    print(f"\nComparing against {len(database_embeddings)} faces in DB...")
    
    min_dist = float('inf')
    best_match = None

    for name, db_emb in database_embeddings.items():
        dist = np.linalg.norm(test_embedding - db_emb)
        if dist < min_dist:
            min_dist = dist
            best_match = name

    print("-" * 30)
    print(f"Best Match: {best_match}")
    print(f"Distance:   {min_dist:.4f}")
    print("-" * 30)

    if min_dist < 0.8:  # Slightly relaxed threshold for testing
        print("✅ MATCH SUCCESS! (Distance < 0.8)")
    else:
        print("❌ STILL NO MATCH. Distance is too high.")
        print("Possible reasons:")
        print("1. The test photo lighting is very different from the CCTV.")
        print("2. The angle (profile vs front) is too extreme.")

if __name__ == "__main__":
    main()