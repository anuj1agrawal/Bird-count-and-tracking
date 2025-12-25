# NOTE:
# The original dataset video is company-restricted and cannot be included in this repository.
# To run the project successfully, please add your dataset video to the project root directory
# with the exact filename: "chicken_dataset_video.MP4".
# If the file is missing or renamed, the pipeline will raise an error during execution

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import shutil
from pathlib import Path
from complete_pipeline_density import process_video_with_density_method
import json

# Initialize FastAPI app
app = FastAPI(
    title="Bird Counting & Weight Estimation API",
    description="Poultry CCTV video analysis with detection, tracking, and weight estimation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create outputs directory
OUTPUT_DIR = Path("api_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Bird Counting & Weight Estimation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "analyze_video": "/analyze_video (POST)",
            "download_video": "/download_video/{filename} (GET)"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "OK",
        "service": "Bird Counting API",
        "model": "YOLOv8n",
        "tracking": "ByteTrack"
    }

@app.post("/analyze_video")
async def analyze_video(
    video: UploadFile = File(..., description="Video file (MP4, AVI, MOV)"),
    fps_sample: float = Form(None, description="Sample every Nth frame (None = all frames)"),
    conf_thresh: float = Form(0.25, description="Detection confidence threshold (0.0-1.0)"),
    iou_thresh: float = Form(0.5, description="IOU threshold for NMS (not used in current impl)")
):
    """
    Analyze poultry CCTV video for bird counting and weight estimation
    
    Args:
        video: Video file upload
        fps_sample: Frame sampling rate (None for all frames)
        conf_thresh: Detection confidence threshold
        iou_thresh: IOU threshold (placeholder)
    
    Returns:
        JSON with counts, tracking data, and weight estimates
    """
    
    # Validate file type
    if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload MP4, AVI, or MOV video."
        )
    
    # Validate parameters
    if not 0.0 <= conf_thresh <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="conf_thresh must be between 0.0 and 1.0"
        )
    
    temp_input = None
    temp_output = None
    
    try:
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            shutil.copyfileobj(video.file, tmp)
            temp_input = tmp.name
        
        # Generate output path
        output_filename = f"annotated_{Path(video.filename).stem}.mp4"
        temp_output = str(OUTPUT_DIR / output_filename)
        
        print(f" Processing video: {video.filename}")
        print(f" Config: conf={conf_thresh}, fps_sample={fps_sample}")
        
        # Process video
        results = process_video_with_density_method(
            source_path=temp_input,
            output_path=temp_output,
            sample_every_n_frames=int(fps_sample) if fps_sample else 1
        )
        
        # Prepare response
        response_data = {
            "status": "success",
            "video_info": results["video_info"],
            "counts": results["counts"],
            "tracks_sample": results["tracks_sample"][:10],  # First 10 tracks
            "weight_estimates": results["weight_estimates"],
            "artifacts": {
                "annotated_video": output_filename,
                "download_url": f"/download_video/{output_filename}"
            },
            "metadata": {
                "input_filename": video.filename,
                "confidence_threshold": conf_thresh,
                "fps_sampling": fps_sample
            }
        }
        
        print(f" Processing complete: {output_filename}")
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        print(f" Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Cleanup temp input file
        if temp_input and os.path.exists(temp_input):
            try:
                os.unlink(temp_input)
            except:
                pass

@app.get("/download_video/{filename}")
async def download_video(filename: str):
    """
    Download annotated video file
    
    Args:
        filename: Name of the output video file
    
    Returns:
        Video file download
    """
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=filename
    )

@app.get("/list_outputs")
async def list_outputs():
    """List all generated output files"""
    files = list(OUTPUT_DIR.glob("*.mp4"))
    return {
        "count": len(files),
        "files": [f.name for f in files]
    }

@app.delete("/clear_outputs")
async def clear_outputs():
    """Clear all output files (for testing/cleanup)"""
    files = list(OUTPUT_DIR.glob("*"))
    for f in files:
        try:
            f.unlink()
        except:
            pass
    return {"message": f"Cleared {len(files)} files"}

if __name__ == "__main__":
    import uvicorn
    print("Starting Bird Counting API...")
    print("API docs available at: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000)