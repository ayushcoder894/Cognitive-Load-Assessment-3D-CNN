"""
MediaPipe 468 Landmark Extractor for Student Videos

This script processes student videos from the specified folder,
extracts 468 MediaPipe face landmarks per frame, and saves them
in CSV format matching the given example.



"""

# Handling missing dependencies in Docker environments
import sys
import os
import platform

# Docker environment dependency check and installation guide
try:
    import cv2
except ImportError as e:
    if "libGL.so.1" in str(e):
        print("=" * 80)
        print("ERROR: Missing OpenCV dependencies (libGL.so.1)")
        print("=" * 80)
        print("\nTo fix this issue, run the following commands in your Docker container:")
        print("\n    apt-get update && apt-get install -y libgl1 libglib2.0-0")
        print("\nIf that doesn't work, try:")
        print("\n    apt-get update && apt-get install -y libgl1-mesa-dev")
        print("\nOr for Ubuntu 24.04 (Noble):")
        print("\n    apt-get update && apt-get install -y libgl1 libglib2.0-0t64")
        print("\nOr try installing the complete OpenCV dependencies:")
        print("\n    apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libpython3-dev python3-opencv")
        print("\nThen try running the script again.")
        print("\nFull error: ", str(e))
        sys.exit(1)
    else:
        raise e

# Now import remaining libraries
import mediapipe as mp
import pandas as pd
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MediaPipe468Extractor:
    """
    Extracts 468 MediaPipe face landmarks from videos and saves in CSV format.
    """
    
    def __init__(self, input_dir, output_dir, max_videos=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.max_videos = max_videos
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Performance tracking
        self.processed_videos = 0
        self.processed_frames = 0
        self.start_time = time.time()
    
    def setup_mediapipe(self):
        """
        Set up MediaPipe Face Mesh with optimal settings for videos
        """
        return self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )
    
    def calculate_ear(self, landmarks):
        """
        Calculate Eye Aspect Ratio (EAR) for both eyes
        Based on MediaPipe face mesh indices for eyes
        """
        # Left eye indices
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        
        # Right eye indices
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        
        # Extract coordinates
        left_eye_points = np.array([landmarks[i] for i in left_eye_indices])
        right_eye_points = np.array([landmarks[i] for i in right_eye_indices])
        
        # Calculate EAR for left eye
        left_ear = self._calculate_single_ear(left_eye_points)
        
        # Calculate EAR for right eye
        right_ear = self._calculate_single_ear(right_eye_points)
        
        # Average EAR
        avg_ear = (left_ear + right_ear) / 2
        
        return left_ear, right_ear, avg_ear
    
    def _calculate_single_ear(self, eye_points):
        """
        Calculate EAR for a single eye
        EAR = (|p2-p6| + |p3-p5|) / (2|p1-p4|)
        """
        if len(eye_points) != 6:
            return 0.0
        
        # Vertical distances
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Horizontal distance
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Calculate EAR
        if C == 0:
            return 0.0
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def process_frame(self, image, face_mesh):
        """
        Process a single frame to extract landmarks
        """
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Process with MediaPipe
        results = face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract normalized landmark coordinates
        landmarks = []
        for landmark in face_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        
        landmarks_array = np.array(landmarks)
        
        # Calculate EAR
        left_ear, right_ear, avg_ear = self.calculate_ear(landmarks_array)
        
        # Extract X and Y coordinates for all 468 landmarks
        x_coords = landmarks_array[:, 0].tolist()
        y_coords = landmarks_array[:, 1].tolist()
        
        return {
            'leftEAR': left_ear,
            'rightEAR': right_ear,
            'EAR': avg_ear,
            'behaviroal_AU_X': x_coords,
            'behaviroal_AU_Y': y_coords
        }
    
    def process_video(self, video_path):
        """
        Process a video to extract face landmarks for each frame
        """
        video_path = Path(video_path)
        print(f"\nProcessing video: {video_path.name}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {width}x{height}, {fps} FPS, {frame_count} frames")
        
        # Initialize MediaPipe
        with self.setup_mediapipe() as face_mesh:
            # Process frames
            frame_results = []
            pbar = tqdm(total=frame_count, desc="Processing frames")
            
            frame_idx = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # Process frame
                frame_data = self.process_frame(frame, face_mesh)
                
                # If face detected
                if frame_data:
                    frame_results.append(frame_data)
                else:
                    # Store empty values when no face detected
                    frame_results.append({
                        'leftEAR': 0.0,
                        'rightEAR': 0.0,
                        'EAR': 0.0,
                        'behaviroal_AU_X': [0.0] * 468,
                        'behaviroal_AU_Y': [0.0] * 468
                    })
                
                frame_idx += 1
                pbar.update(1)
                
                # For testing purposes, limit frames if needed
                # if frame_idx > 100:
                #     break
            
            pbar.close()
            cap.release()
            
            # Create DataFrame
            if frame_results:
                df = pd.DataFrame(frame_results)
                self.processed_frames += len(df)
                return df
            else:
                print(f"No valid frames in {video_path.name}")
                return None
    
    def save_results(self, df, video_path):
        """
        Save the results to CSV
        """
        if df is None or len(df) == 0:
            return False
        
        video_path = Path(video_path)
        # Create student folder if it doesn't exist
        student_id = video_path.parent.name
        student_folder = self.output_dir / student_id
        student_folder.mkdir(exist_ok=True, parents=True)
        
        # Create output filename based on video name
        video_name = video_path.stem
        csv_filename = student_folder / f"{video_name}.csv"
        
        # Save to CSV
        df.to_csv(csv_filename, index=False)
        print(f"✓ Saved: {csv_filename} ({len(df)} frames)")
        
        return True
    
    def process_all_videos(self):
        """
        Process all videos in the input directory
        """
        print("🚀 Starting MediaPipe 468 Landmark Extraction")
        print("=" * 60)
        
        # Find all video files
        video_files = []
        for student_dir in self.input_dir.iterdir():
            if student_dir.is_dir():
                for video_path in student_dir.glob("*.mp4"):
                    video_files.append(video_path)
        
        print(f"Found {len(video_files)} video files across {len(list(self.input_dir.glob('*/'))) - 1} students")
        
        # Process videos
        if self.max_videos:
            video_files = video_files[:self.max_videos]
            print(f"Processing first {self.max_videos} videos for testing")
        
        for video_path in video_files:
            try:
                # Process video
                df = self.process_video(video_path)
                
                # Save results
                if df is not None:
                    success = self.save_results(df, video_path)
                    if success:
                        self.processed_videos += 1
                
            except Exception as e:
                print(f"❌ Error processing {video_path.name}: {str(e)}")
        
        # Print summary
        elapsed_time = time.time() - self.start_time
        print("\n" + "=" * 60)
        print("🎯 EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"✓ Videos processed: {self.processed_videos}/{len(video_files)}")
        print(f"✓ Total frames processed: {self.processed_frames:,}")
        print(f"✓ Time taken: {elapsed_time:.2f} seconds")
        print(f"✓ Average processing speed: {self.processed_frames/elapsed_time:.2f} frames per second")
        print(f"💾 Output directory: {self.output_dir}")
        
        # Create summary file
        summary = {
            'videos_processed': self.processed_videos,
            'videos_total': len(video_files),
            'frames_processed': self.processed_frames,
            'time_taken_seconds': elapsed_time,
            'frames_per_second': self.processed_frames/elapsed_time if elapsed_time > 0 else 0
        }
        
        # Save summary
        summary_file = self.output_dir / "extraction_summary.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"💾 Summary saved: {summary_file}")
        
        return summary

def main():
    """
    Main execution function
    """
    print("=" * 80)
    print("MediaPipe 468 Landmark Extractor (Docker-compatible version)")
    print("=" * 80)
    
    # Check for Docker environment and provide guidance
    if os.path.exists('/.dockerenv'):
        print("\nDetected Docker environment.")
        print("Ubuntu version:", platform.platform())
        print("If you encounter dependency issues, try the following:")
        print("    apt-get update && apt-get install -y libgl1 libglib2.0-0")
        print("    # or")
        print("    apt-get update && apt-get install -y libgl1-mesa-dev")
        print("    # or for Ubuntu 24.04 (Noble)")
        print("    apt-get update && apt-get install -y libgl1 libglib2.0-0t64")
    
    # Define paths - adjust for Docker environments
    default_input_dir = "C:/ML_ED/data/student_data_30fps"
    default_output_dir = "C:/ML_ED/data/student_landmarks_output"
    
    # Allow path override through environment variables or command line args
    input_dir = os.environ.get('INPUT_DIR', default_input_dir)
    output_dir = os.environ.get('OUTPUT_DIR', default_output_dir)
    
    # Handle command line arguments if provided
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create extractor
    extractor = MediaPipe468Extractor(input_dir, output_dir, max_videos=None)
    
    # Process all videos
    summary = extractor.process_all_videos()
    
    print(f"\n🎉 MediaPipe 468 landmark extraction completed!")

if __name__ == "__main__":
    main()

