"""
Extract Frames from CCTV Video
=============================

This module extracts frames from CCTV video footage at specified intervals.
The extracted frames are saved for face detection and feature extraction.

Author: Computer Vision Student
"""

import cv2
import os
import argparse
from pathlib import Path


class FrameExtractor:
    """Extract frames from video files at specified intervals."""
    
    def __init__(self, video_path, output_dir, frame_interval=30):
        """
        Initialize the FrameExtractor.
        
        Args:
            video_path (str): Path to the input video file
            output_dir (str): Directory to save extracted frames
            frame_interval (int): Extract every Nth frame (default: 30)
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.frame_interval = frame_interval
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_frames(self):
        """
        Extract frames from video and save them as images.
        
        Returns:
            int: Number of frames extracted
        """
        print(f"Extracting frames from: {self.video_path}")
        print(f"Saving to: {self.output_dir}")
        print(f"Extracting every {self.frame_interval}th frame...")
        
        # Open video capture
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video FPS: {fps}")
        print(f"Total frames: {total_frames}")
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Extract every Nth frame
            if frame_count % self.frame_interval == 0:
                # Generate filename with zero-padding
                filename = f"frame_{extracted_count:06d}.jpg"
                filepath = self.output_dir / filename
                
                # Save frame
                cv2.imwrite(str(filepath), frame)
                extracted_count += 1
                
                if extracted_count % 10 == 0:
                    print(f"Extracted {extracted_count} frames...")
            
            frame_count += 1
        
        cap.release()
        print(f"\nExtraction complete! Total frames extracted: {extracted_count}")
        return extracted_count


def main():
    """Main function to run frame extraction from command line."""
    parser = argparse.ArgumentParser(description="Extract frames from CCTV video")
    parser.add_argument("--video", default="data/cctv.mp4",
                       help="Path to CCTV video file")
    parser.add_argument("--output", default="data/frames",
                       help="Output directory for frames")
    parser.add_argument("--interval", type=int, default=30,
                       help="Extract every Nth frame (default: 30)")
    
    args = parser.parse_args()
    
    # Create extractor and run
    extractor = FrameExtractor(args.video, args.output, args.interval)
    extractor.extract_frames()


if __name__ == "__main__":
    main()
