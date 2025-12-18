from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import json
from weight_estimator import WeightEstimator

def process_video_complete(source_path, output_path, conf_thresh=0.25, fps_sample=None):
    """
    Complete pipeline: Detection + Tracking + Weight Estimation
    
    Args:
        source_path: Input video path
        output_path: Output annotated video path
        conf_thresh: Detection confidence threshold
        fps_sample: Sample every Nth frame (None = process all frames)
    
    Returns:
        dict with counts, tracks, and weight estimates
    """
    # Load model
    model = YOLO('yolov8n.pt')
    
    # Initialize video
    cap = cv2.VideoCapture(source_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize tracker and weight estimator
    byte_tracker = sv.ByteTrack()
    weight_estimator = WeightEstimator()  # No calibration = proxy values
    
    # Annotators
    box_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color.GREEN)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.7)
    
    # Data storage
    counts_over_time = []
    tracks_sample = []
    frame_count = 0
    processed_count = 0
    
    print(f" Processing video: {total_frames} frames at {fps} FPS")
    print(f" Confidence threshold: {conf_thresh}")
    if fps_sample:
        print(f" Sampling every {fps_sample} frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Frame sampling logic
        if fps_sample and frame_count % fps_sample != 0:
            frame_count += 1
            out.write(frame)  
            continue
        
        # Run detection
        results = model(frame, conf=conf_thresh, classes=[14], verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Update tracker
        detections = byte_tracker.update_with_detections(detections)
        
        # Calculate weights for each detection
        labels = []
        for i, (bbox, tracker_id, confidence) in enumerate(
            zip(detections.xyxy, detections.tracker_id, detections.confidence)
        ):
            # Update weight for this track
            weight_estimator.update_track_weight(tracker_id, bbox)
            weight_proxy, _ = weight_estimator.estimate_weight(bbox)
            
            # Create label
            labels.append(f"ID:{tracker_id} | W:{weight_proxy:.0f}")
            
            # Sample track data (first 5 birds only for JSON)
            if processed_count == 0 and i < 5:
                tracks_sample.append({
                    "id": int(tracker_id),
                    "bbox": [float(x) for x in bbox],
                    "confidence": float(confidence),
                    "weight_proxy": round(float(weight_proxy), 2)
                })
        
        # Annotate frame
        annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        
        # Add overlay info
        bird_count = len(detections)
        timestamp = frame_count / fps
        
        # Info panel
        cv2.rectangle(annotated_frame, (5, 5), (400, 90), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"Count: {bird_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Time: {timestamp:.2f}s", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", (10, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Store count data
        counts_over_time.append({
            "timestamp": round(timestamp, 2),
            "count": bird_count,
            "frame": frame_count
        })
        
        # Write frame
        out.write(annotated_frame)
        processed_count += 1
        frame_count += 1
        
        # Progress update
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames ({frame_count*100/total_frames:.1f}%)")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Get weight summary
    weight_summary = weight_estimator.get_all_weights_summary()
    
    # Prepare results
    results = {
        "video_info": {
            "total_frames": frame_count,
            "fps": fps,
            "duration_seconds": round(frame_count / fps, 2)
        },
        "counts": counts_over_time,
        "tracks_sample": tracks_sample,
        "weight_estimates": weight_summary,
        "artifacts": {
            "annotated_video": output_path,
            "json_output": output_path.replace('.mp4', '_results.json')
        }
    }
    
    # Save JSON
    json_path = output_path.replace('.mp4', '_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f" Video: {output_path}")
    print(f" JSON: {json_path}")
    print(f" Average bird count: {np.mean([c['count'] for c in counts_over_time]):.1f}")
    print(f"  Average weight proxy: {weight_summary['avg_weight']}")
    print(f" Total unique birds tracked: {weight_summary['total_birds']}")
    
    return results


if __name__ == "__main__":
    # Configuration
    SOURCE_VIDEO = "chicken_dataset_video.mp4" 
    OUTPUT_VIDEO = "outputs/final_annotated.mp4"
    
    # Process video
    results = process_video_complete(
        source_path=SOURCE_VIDEO,
        output_path=OUTPUT_VIDEO,
        conf_thresh=0.30,
        fps_sample=None  
    )
    
    # Display sample results
    print("\n" + "="*50)
    print(" SAMPLE RESULTS")
    print("="*50)
    print(f"\n First 5 timestamps:")
    for c in results['counts'][:5]:
        print(f"  {c['timestamp']}s → {c['count']} birds")
    
    print(f"\n Sample tracked birds:")
    for t in results['tracks_sample'][:3]:
        print(f"  ID {t['id']}: weight_proxy={t['weight_proxy']}, conf={t['confidence']:.2f}")

    print(f"\n Weight Statistics:")
    ws = results['weight_estimates']
    print(f"  Average: {ws['avg_weight']} {ws['unit']}")
    print(f"  Range: {ws['min_weight']} - {ws['max_weight']} {ws['unit']}")
    print(f"  Birds tracked: {ws['total_birds']}")