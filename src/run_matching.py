import torch
import cv2
import os
import pickle
import argparse
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision import transforms

# --- CONFIG ---
DB_PATH = 'database/embeddings.pkl'
DEFAULT_TEST_IMG = 'test.jpg'
TEMP_CROP_PATH = 'temp_test_crop.jpg'  # Temporary file for consistency

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=DEFAULT_TEST_IMG, help="Path to test image")
    parser.add_argument("--threshold", type=float, default=0.7, help="Distance threshold (lower is stricter)")
    args = parser.parse_args()

    print("--- CCTV MATCHING SYSTEM ---")

    # 1. Load Database
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database not found at {DB_PATH}")
        print("Run src/generate_embeddings.py first.")
        return

    with open(DB_PATH, 'rb') as f:
        database = pickle.load(f)
        
    if 'embeddings' in database:
        database = {k: v['embedding'] for k, v in database['embeddings'].items()}

    print(f"Loaded {len(database)} identities from database.")

    # 2. Prepare AI Models
    print("Loading AI Models...")
    mtcnn = MTCNN(keep_all=False, select_largest=True, device='cpu')
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 3. Process Test Image
    if not os.path.exists(args.test):
        print(f"❌ Error: Test image '{args.test}' not found.")
        return

    img = Image.open(args.test).convert('RGB')
    
    # --- THE MAGIC FIX: Save and Reload ---
    # We save the crop to disk and reload it. This ensures the test image 
    # has the EXACT same format/compression as the database images.
    
    mtcnn(img, save_path=TEMP_CROP_PATH)
    
    if not os.path.exists(TEMP_CROP_PATH):
        print("❌ No face detected in the test image.")
        return

    # Reload the crop from disk
    img_crop = Image.open(TEMP_CROP_PATH).convert('RGB')
    img_tensor = transform(img_crop).unsqueeze(0)

    # Generate Embedding
    with torch.no_grad():
        test_embedding = resnet(img_tensor).detach().numpy()[0]

    # Clean up temp file
    try:
        os.remove(TEMP_CROP_PATH)
    except:
        pass

    # 4. Find Best Match
    min_dist = float('inf')
    best_match_name = None

    for name, db_embedding in database.items():
        dist = np.linalg.norm(test_embedding - db_embedding)
        if dist < min_dist:
            min_dist = dist
            best_match_name = name

    # 5. Result
    print("-" * 30)
    print(f"Test Image:   {args.test}")
    print(f"Best Match:   {best_match_name}")
    print(f"Distance:     {min_dist:.4f}")
    print("-" * 30)

    if min_dist < args.threshold:
        print("✅ MATCH CONFIRMED!")
        print(f"Identity verified as: {best_match_name}")
    else:
        print("❌ NO MATCH FOUND.")
        print("Distance is too high (Likely different people).")

if __name__ == "__main__":
    main()