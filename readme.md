**README: YOLO Tracking System**

**1. Overview**

This project implements an object detection and tracking system using YOLO for object detection and BYTETracker for multi-object tracking. The script processes a given video file to detect objects in each frame and tracks their movements across frames.

**2. Requirements**

- Python 3.x
- OpenCV
- Ultralytics YOLO
- NumPy

You can install the required dependencies using the following command:

```
pip install opencv-python ultralytics numpy
```

**3. Running the Script**

To run the tracking system, use the following parameters:

- `video_path`: Path to the input video file. For example: `demo.mp4`
- `model_path`: Path to the YOLO model weights file. For example: `best.pt`
- `tracker_cfg`: Dictionary containing configuration parameters for BYTETracker.

Example of running the tracking function:

```python
run_tracking(video_path, model_path, tracker_cfg)
```

**4. Configuration**

You can customize BYTETracker by modifying `tracker_cfg` parameters:

- `tracker_type`: The type of tracker used, set to `'bytetrack'`.
- `track_high_thresh`: High confidence threshold for track initialization.
- `track_low_thresh`: Low confidence threshold for potential tracks.
- `new_track_thresh`: Threshold to initialize a new track.
- `track_buffer`: The maximum number of frames a track can remain inactive.
- `match_thresh`: Threshold for matching detections with existing tracks.
- `fuse_score`: Boolean to enable score fusion.

**5. Usage**

- Once you run the script, it will open a window displaying the video with detected and tracked objects marked by rectangles and IDs.
- You can exit the tracking by pressing the "q" key.

**6. Notes**

- Ensure the model weights (`best.pt`) are compatible with the YOLO version being used.
- Adjust tracking parameters to suit your specific use case for optimal performance.

**7. License**

This project is provided under an MIT license. You can use, modify, and distribute it freely.

