from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
from collections import defaultdict
import os

def process_video_with_tracking(source_path, output_path):
    """
    Process video with bird detection and tracking
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load model
    model = YOLO('yolov8n.pt')
    
    # Initialize video capture
    cap = cv2.VideoCapture(source_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize ByteTrack tracker
    byte_tracker = sv.ByteTrack()
    
    # Annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.8)
    
    # Data collection
    counts_over_time = []
    frame_count = 0
    
    print(f"Processing video: {total_frames} frames at {fps} FPS")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection (bird class only)
        results = model(frame, conf=0.25, classes=[14], verbose=False)
        
        # Convert to supervision format
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Update tracker
        detections = byte_tracker.update_with_detections(detections)
        
        # Create labels with tracking IDs
        labels = [
            f"ID:{tracker_id} {confidence:.2f}"
            for tracker_id, confidence in zip(detections.tracker_id, detections.confidence)
        ]
        
        # Annotate frame
        annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        
        # Add count overlay
        bird_count = len(detections)
        timestamp = frame_count / fps
        cv2.putText(annotated_frame, f"Count: {bird_count} | Time: {timestamp:.1f}s", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save count data
        counts_over_time.append({
            "timestamp": round(timestamp, 2),
            "count": bird_count,
            "frame": frame_count
        })
        
        # Write frame
        out.write(annotated_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count*100/total_frames:.1f}%)")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f" Processing complete! Output saved to: {output_path}")
    print(f"Total frames processed: {frame_count}")
    print(f"Average bird count: {np.mean([c['count'] for c in counts_over_time]):.1f}")
    
    return {
        "counts": counts_over_time,
        "total_frames": frame_count,
        "fps": fps
    }

# Test the pipeline
if __name__ == "__main__":
    source = "chicken_dataset_video.mp4"  
    output = "outputs/annotated_tracking.mp4"
    
    results = process_video_with_tracking(source, output)
    
    print("\n Sample counts:")
    for c in results["counts"][:5]:
        print(f"  Time {c['timestamp']}s: {c['count']} birds")