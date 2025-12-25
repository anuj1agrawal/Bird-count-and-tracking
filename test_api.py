# NOTE:
# The original dataset video is company-restricted and cannot be included in this repository.
# To run the project successfully, please add your dataset video to the project root directory
# with the exact filename: "chicken_dataset_video.MP4".
# If the file is missing or renamed, the pipeline will raise an error during execution



import requests
import time
import subprocess
import os
from pathlib import Path

# Start the server in background
print("Starting API server...")
server_process = subprocess.Popen(
    ["python", "main.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=os.getcwd()
)

# Give a moment for immediate crash detection
time.sleep(2)

# If server crashed immediately, show error
if server_process.poll() is not None:
    stdout, stderr = server_process.communicate()
    print("STDOUT:\n", stdout.decode())
    print("STDERR:\n", stderr.decode())
    raise RuntimeError("Server crashed on startup")

# Wait until health endpoint is available
print("Waiting for server to become ready...")
for _ in range(20):
    try:
        response = requests.get("http://localhost:8000/health", timeout=1)
        if response.status_code == 200:
            break
    except:
        time.sleep(1)
else:
    server_process.terminate()
    raise RuntimeError("API server did not start")

try:
    # Test 1: Health endpoint
    print("\nTest 1: Health Check")
    print("-" * 50)
    response = requests.get("http://localhost:8000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

    # Test 2: Root endpoint
    print("\nTest 2: Root Endpoint")
    print("-" * 50)
    response = requests.get("http://localhost:8000/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

    # Test 3: Video analysis (only if video exists)
    video_path = Path("chicken_dataset_video.mp4")
    if video_path.exists():
        print("\nTest 3: Video Analysis")
        print("-" * 50)
        with open(video_path, "rb") as f:
            files = {"video": f}
            data = {"conf_thresh": 0.30}
            response = requests.post(
                "http://localhost:8000/analyze_video",
                files=files,
                data=data
            )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    else:
        print("\nTest 3 skipped: video file not found")

    print("\nAll tests completed successfully")

except Exception as e:
    print("Error during testing:", e)

finally:
    # Stop the server
    print("\nStopping server...")
    server_process.terminate()
    server_process.wait()
    print("Server stopped.")
