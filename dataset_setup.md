# Dataset Setup Instructions

## Overview

This project was developed and tested using the official Kuppismart poultry farm CCTV dataset. Due to data privacy and proprietary restrictions, the dataset video is not included in this submission package.

## Required Dataset

**Video Specifications**:
- Format: MP4, AVI, or MOV
- Content: Overhead poultry farm CCTV footage
- Minimum duration: 10 seconds recommended
- Resolution: Any (tested on 1920x1080)
- Frame rate: Any (tested on 19 FPS)

## Setup Instructions

### Option 1: Using Provided Dataset (For Kuppismart Team)

1. Navigate to the project's `videos/` directory:
   ```bash
   cd videos/
   ```

2. Place the official dataset video and rename it to:
   ```
   chicken_dataset_video.mp4
   ```
   
   **Exact filename required**: `chicken_dataset_video.mp4`

3. Verify the file is in place:
   ```bash
   ls videos/
   # Should show: chicken_dataset_video.mp4
   ```

4. Return to project root and run:
   ```bash
   python complete_pipeline_density.py
   ```

### Option 2: Using Custom Video File

If you want to use a different filename or test with your own video:

1. Place your video in `videos/` directory with any name

2. Update the source path in `complete_pipeline_density.py`:
   ```python
   # Line 300
   SOURCE_VIDEO = "videos/your_video_filename.mp4"
   ```

3. Run the pipeline:
   ```bash
   python complete_pipeline_density.py
   ```

### Option 3: Using FastAPI Service (No Filename Restriction)

The API accepts any video filename via file upload:

1. Start the API server:
   ```bash
   python main.py
   ```

2. Upload any video via the API:
   ```bash
   curl -X POST "http://localhost:8000/analyze_video" \
     -F "video=@path/to/your/video.mp4"
   ```

3. Or use the interactive docs at `http://localhost:8000/docs`

## Dataset Access

### For Kuppismart Evaluation Team

The official dataset used for testing was provided via:
- **Source**: Google Drive link shared in candidate task email
- **Date**: December 23, 2025
- **Folder**: Contains `Poultry_Sample.mp4` and related files

If you need to re-access the dataset for evaluation purposes, please refer to the original Google Drive link provided in the task assignment email.

### For External Users / Future Development

If you wish to test this system with your own poultry farm footage:

1. **Recording Setup**: 
   - Fixed overhead camera angle
   - Good lighting conditions
   - Clear view of farm floor
   - Minimum 720p resolution recommended

2. **Video Requirements**:
   - MP4 format preferred
   - H.264 codec recommended
   - 10+ FPS minimum
   - 10+ seconds duration

3. **Sample Datasets** (Public alternatives):
   - Roboflow Universe: Search "chicken detection" or "poultry farm"
   - YouTube: Search "poultry farm CCTV" (use with permission)
   - Academic datasets: Check computer vision research repositories

## Troubleshooting

### Error: "Video file not found"

**Cause**: The system cannot locate `videos/chicken_dataset_video.mp4`

**Solution**:
1. Verify the file exists:
   ```bash
   ls -la videos/chicken_dataset_video.mp4
   ```
2. Check filename exactly matches (case-sensitive)
3. Ensure file is not corrupted (try playing it in VLC)

### Error: "Cannot open video file"

**Cause**: Video codec or format not supported

**Solution**:
1. Convert video to standard MP4 using ffmpeg:
   ```bash
   ffmpeg -i input.mp4 -c:v libx264 -c:a aac videos/chicken_dataset_video.mp4
   ```
2. Verify video plays in standard media player

### Error: "No detections found"

**Cause**: Video content doesn't match expected poultry farm footage

**Solution**:
1. Verify video contains visible chickens
2. Check if camera angle is overhead or angled downward
3. Adjust detection parameters in code if needed

## Performance Notes

**Tested Dataset Characteristics**:
- Duration: ~5 minutes (291 seconds)
- Total frames: 5,545
- Frame rate: 19 FPS
- Resolution: 1920x1080
- Content: ~300-400 white chickens in dense indoor farm
- Lighting: Artificial indoor lighting, consistent

**Processing Performance**:
- Processing speed: ~19 FPS (real-time)
- Total processing time: ~5 minutes for 5-minute video
- Memory usage: ~500MB RAM
- GPU: Not required (CPU-only processing)

## Contact

For dataset-related questions or access issues, please contact:
- Kuppismart Solutions Pvt Ltd
- Email: [As provided in task documentation]

---

**Note**: This documentation assumes the evaluator has access to the original dataset shared during the candidate task assignment. All code is configured to work with that specific dataset out of the box.
