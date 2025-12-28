import cv2
import torch
import os
from pathlib import Path
from facenet_pytorch import MTCNN
from PIL import Image

class FaceDetector:
    """
    Detect faces in extracted frames using MTCNN (via facenet-pytorch).
    This replaces the older OpenCV DNN method to avoid download errors.
    """
    
    def __init__(self, frames_dir, faces_dir, confidence_threshold=0.9):
        self.frames_dir = Path(frames_dir)
        self.faces_dir = Path(faces_dir)
        # MTCNN uses 'prob' for threshold, usually requires high confidence
        self.confidence_threshold = confidence_threshold
        
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MTCNN
        # keep_all=False means we only take the most prominent face
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading MTCNN on device: {device}")
        self.mtcnn = MTCNN(keep_all=False, select_largest=True, device=device)
        
    def detect_faces(self):
        print(f"Detecting faces in: {self.frames_dir}")
        print(f"Saving faces to: {self.faces_dir}")
        
        image_files = list(self.frames_dir.glob("*.jpg")) + \
                     list(self.frames_dir.glob("*.png")) + \
                     list(self.frames_dir.glob("*.jpeg"))
        
        if not image_files:
            print(f"No image files found in {self.frames_dir}")
            return 0
            
        total_faces = 0
        
        for i, image_file in enumerate(image_files):
            try:
                # MTCNN expects PIL images
                img = Image.open(image_file)
                
                # Detect and crop
                # save_path automatically saves the file if a face is found
                save_filename = f"{image_file.stem}_face.jpg"
                save_path = self.faces_dir / save_filename
                
                # MTCNN returns the cropped tensor, but we use the save_path feature
                # which saves it directly to disk
                self.mtcnn(img, save_path=str(save_path))
                
                if save_path.exists():
                    total_faces += 1
            except Exception as e:
                print(f"Skipping {image_file.name}: {e}")
                
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_files)} frames...")

        print(f"\nFace detection complete! Total faces detected: {total_faces}")
        return total_faces

if __name__ == "__main__":
    # Test run
    detector = FaceDetector("data/frames", "data/faces")
    detector.detect_faces()