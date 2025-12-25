"""
Complete Pipeline with Density-Based Counting
Optimized for small, dense overhead poultry footage
"""

import cv2
import numpy as np
import json
from collections import defaultdict

def count_and_detect_chickens(frame):
    """
    Detect and count chickens using image processing
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to preserve edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Adaptive threshold for white chickens on brown floor
    thresh = cv2.adaptiveThreshold(
        filtered, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, -8  # Negative to capture lighter regions
    )
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and collect detections
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area (adjust based on chicken size in video)
        if 150 < area < 2500:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter by aspect ratio (chickens are roughly circular)
            if 0.4 < aspect_ratio < 2.5:
                detection = {
                    'bbox': [x, y, x+w, y+h],
                    'area': area,
                    'confidence': 0.8  # Fixed confidence for density method
                }
                detections.append(detection)
    
    return detections

def simple_tracking(current_detections, previous_tracks, max_distance=100):
    """
    Simple tracking based on position proximity with track persistence
    """
    if not previous_tracks:
        # Initialize tracks
        tracks = []
        for i, det in enumerate(current_detections):
            tracks.append({
                'id': i + 1,
                'bbox': det['bbox'],
                'age': 1,
                'frames_lost': 0
            })
        return tracks
    
    # Match current detections to previous tracks
    used_detections = set()
    updated_tracks = []
    next_id = max([t['id'] for t in previous_tracks]) + 1 if previous_tracks else 1
    
    for track in previous_tracks:
        # Skip tracks that have been lost for too long
        if track.get('frames_lost', 0) > 30:
            continue
            
        prev_center = [
            (track['bbox'][0] + track['bbox'][2]) / 2,
            (track['bbox'][1] + track['bbox'][3]) / 2
        ]
        
        best_match = None
        best_distance = max_distance
        
        for i, det in enumerate(current_detections):
            if i in used_detections:
                continue
            
            curr_center = [
                (det['bbox'][0] + det['bbox'][2]) / 2,
                (det['bbox'][1] + det['bbox'][3]) / 2
            ]
            
            distance = np.sqrt(
                (curr_center[0] - prev_center[0])**2 + 
                (curr_center[1] - prev_center[1])**2
            )
            
            if distance < best_distance:
                best_distance = distance
                best_match = i
        
        if best_match is not None:
            # Update track
            updated_tracks.append({
                'id': track['id'],
                'bbox': current_detections[best_match]['bbox'],
                'age': track['age'] + 1,
                'frames_lost': 0
            })
            used_detections.add(best_match)
        else:
            # Keep track but mark as lost
            updated_tracks.append({
                'id': track['id'],
                'bbox': track['bbox'],
                'age': track['age'] + 1,
                'frames_lost': track.get('frames_lost', 0) + 1
            })
    
    # Create new tracks for unmatched detections
    for i, det in enumerate(current_detections):
        if i not in used_detections:
            updated_tracks.append({
                'id': next_id,
                'bbox': det['bbox'],
                'age': 1,
                'frames_lost': 0
            })
            next_id += 1
    
    return updated_tracks

def estimate_weight_from_detection(bbox):
    """
    Estimate weight proxy from bounding box
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    area = width * height
    height_factor = height / 100.0
    weight_proxy = area * height_factor
    return weight_proxy

def process_video_with_density_method(source_path, output_path, sample_every_n_frames=1):
    """
    Complete pipeline with density-based detection and tracking
    """
    cap = cv2.VideoCapture(source_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Data storage
    counts_over_time = []
    previous_tracks = []
    all_track_weights = defaultdict(list)
    tracks_sample = []
    frame_count = 0
    
    print(f" Processing video: {total_frames} frames at {fps} FPS")
    print(f" Method: Density-based detection + simple tracking")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames if specified
        if frame_count % sample_every_n_frames != 0:
            out.write(frame)
            frame_count += 1
            continue
        
        # Detect chickens
        detections = count_and_detect_chickens(frame)
        
        # Track chickens
        tracks = simple_tracking(detections, previous_tracks, max_distance=100)
        previous_tracks = tracks
        
        # Annotate frame
        annotated = frame.copy()
        
        for track in tracks:
            x1, y1, x2, y2 = [int(c) for c in track['bbox']]
            track_id = track['id']
            
            # Calculate weight proxy
            weight_proxy = estimate_weight_from_detection(track['bbox'])
            all_track_weights[track_id].append(weight_proxy)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"ID:{track_id} | W:{weight_proxy:.0f}"
            cv2.putText(annotated, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Collect sample track data
            if frame_count == 0 and len(tracks_sample) < 10:
                tracks_sample.append({
                    "id": int(track_id),
                    "bbox": [float(c) for c in track['bbox']],
                    "confidence": 0.8,
                    "weight_proxy": round(float(weight_proxy), 2)
                })
        
        # Add info overlay
        bird_count = len(tracks)
        timestamp = frame_count / fps
        
        cv2.rectangle(annotated, (5, 5), (400, 90), (0, 0, 0), -1)
        cv2.putText(annotated, f"Count: {bird_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f"Time: {timestamp:.2f}s", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f"Frame: {frame_count}/{total_frames}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Store count data
        counts_over_time.append({
            "timestamp": round(timestamp, 2),
            "count": bird_count,
            "frame": frame_count
        })
        
        # Write frame
        out.write(annotated)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames ({frame_count*100/total_frames:.1f}%)")
    
    cap.release()
    out.release()
    
    # Calculate weight statistics
    per_bird_weights = []
    for track_id, weights in all_track_weights.items():
        avg_weight = np.mean(weights)
        per_bird_weights.append({
            "id": int(track_id),
            "weight_proxy": round(float(avg_weight), 2),
            "samples": len(weights),
            "std_dev": round(float(np.std(weights)), 2)
        })
    
    avg_weights = [pb['weight_proxy'] for pb in per_bird_weights]
    
    weight_summary = {
        "unit": "proxy",
        "avg_weight": round(float(np.mean(avg_weights)), 2) if avg_weights else 0,
        "min_weight": round(float(np.min(avg_weights)), 2) if avg_weights else 0,
        "max_weight": round(float(np.max(avg_weights)), 2) if avg_weights else 0,
        "per_bird": sorted(per_bird_weights, key=lambda x: x['id'])[:20],  # First 20
        "total_birds": len(per_bird_weights),
        "calibration_needed": True
    }
    
    # Prepare results
    results = {
        "detection_method": "density_based_image_processing",
        "note": "Density-based method used due to small object size in overhead footage. YOLOv8 generic 'bird' class could not detect these tiny chickens.",
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
    
    avg_count = np.mean([c['count'] for c in counts_over_time])
    
    print(f"\n Processing complete!")
    print(f" Video: {output_path}")
    print(f" JSON: {json_path}")
    print(f" Average bird count: {avg_count:.1f}")
    print(f"  Average weight proxy: {weight_summary['avg_weight']}")
    print(f" Total unique birds tracked: {weight_summary['total_birds']}")
    
    return results

if __name__ == "__main__":
    # Configuration
    SOURCE_VIDEO = "chicken_dataset_video.MP4"  # Required: Place your video with this exact name
    OUTPUT_VIDEO = "outputs/final_annotated_density.mp4"
    
    # Process video
    results = process_video_with_density_method(
        source_path=SOURCE_VIDEO,
        output_path=OUTPUT_VIDEO,
        sample_every_n_frames=1  
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