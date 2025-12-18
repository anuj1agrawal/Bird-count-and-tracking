import requests
import time
import subprocess
import os
from pathlib import Path

# Start the server in background
print(" Starting API server...")
server_process = subprocess.Popen(["python", "main.py"], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE,
                                  cwd=os.getcwd())

# Wait for server to start
time.sleep(3)

try:
    # Test 1: Health endpoint
    print("\n Test 1: Health Check")
    print("-" * 50)
    response = requests.get("http://localhost:8000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test 2: Root endpoint
    print("\n Test 2: Root Endpoint")
    print("-" * 50)
    response = requests.get("http://localhost:8000/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test 3: Video analysis (if video exists)
    if Path("chicken_dataset_video.mp4").exists():
        print("\n Test 3: Video Analysis")
        print("-" * 50)
        with open("chicken_dataset_video.mp4", "rb") as f:
            files = {"video": f}
            data = {"conf_thresh": 0.30}
            response = requests.post("http://localhost:8000/analyze_video", 
                                    files=files, data=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    
    print("\n All tests completed!")
    
except Exception as e:
    print(f" Error: {e}")

finally:
    # Kill the server
    print("\n Stopping server...")
    server_process.terminate()
    server_process.wait()
    print("Server stopped.")
