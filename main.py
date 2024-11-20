import cv2
from ultralytics import YOLO
import numpy as np
from ultralytics.trackers import BYTETracker
from ultralytics.utils import IterableSimpleNamespace

def run_tracking(video_path, model_path, tracker_cfg):
    """
    Runs object detection and tracking on a given video file.
    
    Args:
        video_path (str): Path to the input video file.
        model_path (str): Path to the YOLO model weights file.
        tracker_cfg (dict): Configuration dictionary for BYTETracker.
    """
    # Load YOLO model
    model = YOLO(model_path)

    # Configure BYTETracker
    cfg = IterableSimpleNamespace(**tracker_cfg)
    tracker = BYTETracker(args=cfg, frame_rate=30)

    def process_results_with_tracker(results, frame, tracker):
        """
        Extracts detections and interacts with the tracker to update tracking results.

        Args:
            results: Detection results from YOLO model.
            frame (numpy.ndarray): The current video frame.
            tracker (BYTETracker): Instance of the BYTETracker.

        Returns:
            List of tracks with updated tracking results.
        """
        # Extract detection boxes
        det = results.boxes.cpu().numpy() if results.boxes is not None else np.empty((0, 6))
        if len(det) == 0:
            return []  # No detections, return empty tracking results

        # Update tracker with detections
        tracks = tracker.update(det, img=frame)
        if len(tracks) == 0:
            return []  # No tracking results

        return tracks

    # Open video file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # Get detection results from YOLO
            results = model.predict(frame)[0]  # Get the first element from results

            # Process results with tracker
            tracks = process_results_with_tracker(results, frame, tracker)

            # Draw tracking boxes and IDs on the frame
            for track in tracks:
                x_min, y_min, x_max, y_max, track_id = track[:5]  # Extract bounding box and ID
                color = (0, 255, 0)  # Green color
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)  # Draw rectangle
                cv2.putText(frame, f"ID: {int(track_id)}", (int(x_min), int(y_min) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Draw ID text

            # Display the annotated frame
            cv2.imshow("YOLO Tracking", frame)

            # Exit loop if "q" is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


# Example parameters for running the tracking system
video_path = "demo.mp4"
model_path = "best.pt"
tracker_cfg = {
    'tracker_type': 'bytetrack',
    'track_high_thresh': 0.25,
    'track_low_thresh': 0.1,
    'new_track_thresh': 0.25,
    'track_buffer': 30,
    'match_thresh': 0.8,
    'fuse_score': True
}

# Run the tracking system
run_tracking(video_path, model_path, tracker_cfg)
