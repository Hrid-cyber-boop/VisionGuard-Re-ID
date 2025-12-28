"""
Face Matching Module
====================

This module matches test faces against the database of known faces using
CNN embeddings and similarity metrics. It calculates distances between
embeddings to determine if faces belong to the same person.

Author: Computer Vision Student
"""

import cv2
import pickle
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Dict


class FaceMatcher:
    """Match faces against database using CNN embeddings."""
    
    def __init__(self, database_path):
        """
        Initialize the FaceMatcher.
        
        Args:
            database_path (str): Path to embeddings database
        """
        self.database_path = Path(database_path)
        self.embeddings_db = self._load_database()
        
    def _load_database(self):
        """
        Load embeddings database from file.
        
        Returns:
            dict: Loaded database
        """
        if not self.database_path.exists():
            raise ValueError(f"Database not found: {self.database_path}")
        
        print(f"Loading database from: {self.database_path}")
        
        with open(self.database_path, 'rb') as f:
            database = pickle.load(f)
        
        print(f"Loaded {database['total_faces']} faces")
        print(f"Embedding dimension: {database['embedding_dim']}")
        
        return database
    
    def _calculate_euclidean_distance(self, embedding1: np.ndarray, 
                                    embedding2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two embeddings.
        
        Args:
            embedding1 (np.ndarray): First embedding
            embedding2 (np.ndarray): Second embedding
            
        Returns:
            float: Euclidean distance
        """
        return np.linalg.norm(embedding1 - embedding2)
    
    def _calculate_cosine_distance(self, embedding1: np.ndarray, 
                                 embedding2: np.ndarray) -> float:
        """
        Calculate cosine distance between two embeddings.
        
        Args:
            embedding1 (np.ndarray): First embedding
            embedding2 (np.ndarray): Second embedding
            
        Returns:
            float: Cosine distance (1 - cosine similarity)
        """
        # Calculate cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        cosine_similarity = dot_product / (norm1 * norm2 + 1e-10)
        
        # Convert to distance (0 = identical, 2 = opposite)
        cosine_distance = 1 - cosine_similarity
        
        return cosine_distance
    
    def _calculate_distance(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray, 
                          metric: str = 'cosine') -> float:
        """
        Calculate distance between embeddings using specified metric.
        
        Args:
            embedding1 (np.ndarray): First embedding
            embedding2 (np.ndarray): Second embedding
            metric (str): Distance metric ('cosine' or 'euclidean')
            
        Returns:
            float: Distance value
        """
        if metric == 'cosine':
            return self._calculate_cosine_distance(embedding1, embedding2)
        elif metric == 'euclidean':
            return self._calculate_euclidean_distance(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown distance metric: {metric}")
    
    def match_face(self, test_embedding: np.ndarray, 
                   threshold: float = 0.6,
                   metric: str = 'cosine') -> Tuple[bool, List[Dict]]:
        """
        Match a test face against the database.
        
        Args:
            test_embedding (np.ndarray): Embedding of the test face
            threshold (float): Distance threshold for matching
            metric (str): Distance metric to use
            
        Returns:
            Tuple[bool, List[Dict]]: (is_match, list of top matches)
        """
        matches = []
        
        # Compare against all faces in database
        for filename, face_data in self.embeddings_db['embeddings'].items():
            db_embedding = face_data['embedding']
            
            # Calculate distance
            distance = self._calculate_distance(
                test_embedding, db_embedding, metric
            )
            
            matches.append({
                'filename': filename,
                'distance': distance,
                'shape': face_data['shape']
            })
        
        # Sort by distance (closest matches first)
        matches.sort(key=lambda x: x['distance'])
        
        # Check if best match is below threshold
        is_match = len(matches) > 0 and matches[0]['distance'] < threshold
        
        return is_match, matches
    
    def generate_embedding_from_image(self, image_path: str) -> np.ndarray:
        """
        Generate embedding from a test image.
        
        Args:
            image_path (str): Path to test image
            
        Returns:
            np.ndarray: Generated embedding
        """
        # This is a simplified version - in practice, you'd use the same
        # embedding generation as in generate_embeddings.py
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Simple feature extraction (fallback method)
        # In production, use the same FaceNet model as before
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        
        # Extract HOG features
        win_size = (64, 64)
        cell_size = (8, 8)
        block_size = (16, 16)
        block_stride = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, 
                               cell_size, nbins)
        features = hog.compute(gray)
        
        # Normalize
        embedding = features.flatten()
        embedding = embedding / np.linalg.norm(embedding)
        
        # Pad or truncate to match database embedding size
        target_size = self.embeddings_db['embedding_dim']
        if len(embedding) < target_size:
            embedding = np.pad(embedding, (0, target_size - len(embedding)))
        elif len(embedding) > target_size:
            embedding = embedding[:target_size]
        
        return embedding
    
    def visualize_match(self, test_image_path: str, 
                       best_match_filename: str,
                       output_path: str = None):
        """
        Visualize the match between test image and database image.
        
        Args:
            test_image_path (str): Path to test image
            best_match_filename (str): Filename of best match in database
            output_path (str): Path to save visualization
        """
        # Read test image
        test_image = cv2.imread(test_image_path)
        if test_image is None:
            print(f"Could not read test image: {test_image_path}")
            return
        
        # Read best match image (from faces directory)
        match_path = Path("data/faces") / best_match_filename
        match_image = cv2.imread(str(match_path))
        
        if match_image is None:
            print(f"Could not read match image: {match_path}")
            return
        
        # Resize images to same height
        height = min(test_image.shape[0], match_image.shape[0])
        test_image = cv2.resize(test_image, (int(test_image.shape[1] * height / test_image.shape[0]), height))
        match_image = cv2.resize(match_image, (int(match_image.shape[1] * height / match_image.shape[0]), height))
        
        # Concatenate images side by side
        comparison = np.concatenate([test_image, match_image], axis=1)
        
        # Add labels
        cv2.putText(comparison, "Test Image", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(comparison, f"Best Match: {best_match_filename}", 
                   (test_image.shape[1] + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display or save
        if output_path:
            cv2.imwrite(output_path, comparison)
            print(f"Visualization saved to: {output_path}")
        else:
            cv2.imshow("Face Match", comparison)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    """Main function to run face matching from command line."""
    parser = argparse.ArgumentParser(description="Match faces against database")
    parser.add_argument("--test", required=True,
                       help="Path to test image")
    parser.add_argument("--database", default="database/embeddings.pkl",
                       help="Path to embeddings database")
    parser.add_argument("--threshold", type=float, default=0.6,
                       help="Distance threshold for matching")
    parser.add_argument("--metric", default="cosine",
                       choices=["cosine", "euclidean"],
                       help="Distance metric to use")
    parser.add_argument("--visualize", action="store_true",
                       help="Visualize the match")
    
    args = parser.parse_args()
    
    # Create matcher
    matcher = FaceMatcher(args.database)
    
    # Generate embedding for test image
    print(f"Processing test image: {args.test}")
    test_embedding = matcher.generate_embedding_from_image(args.test)
    
    # Match against database
    is_match, matches = matcher.match_face(
        test_embedding, 
        threshold=args.threshold, 
        metric=args.metric
    )
    
    # Display results
    print(f"\n{'='*50}")
    print("MATCHING RESULTS")
    print(f"{'='*50}")
    
    if is_match:
        print(f"✓ MATCH FOUND!")
        print(f"Best match: {matches[0]['filename']}")
        print(f"Distance ({args.metric}): {matches[0]['distance']:.4f}")
    else:
        print(f"✗ NO MATCH FOUND")
        if matches:
            print(f"Closest match: {matches[0]['filename']}")
            print(f"Distance ({args.metric}): {matches[0]['distance']:.4f}")
            print(f"(Above threshold of {args.threshold})")
    
    print(f"\nTop 5 closest matches:")
    for i, match in enumerate(matches[:5]):
        print(f"{i+1}. {match['filename']:<30} distance: {match['distance']:.4f}")
    
    # Visualize if requested
    if args.visualize and matches:
        matcher.visualize_match(args.test, matches[0]['filename'])


if __name__ == "__main__":
    main()
