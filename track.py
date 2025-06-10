import cv2
import numpy as np
from sort import Sort

# Function to read detections from detections.txt file
def read_detections(detection_file):
    detections = {}
    with open(detection_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue  # Skip lines that don't have enough data
            frame_id, _, x, y, w, h, confidence = map(float, parts[:7])
            x1, y1, x2, y2 = x, y, x + w, y + h  # Convert width and height to x2, y2
            if frame_id not in detections:
                detections[frame_id] = []
            detections[frame_id].append((x1, y1, x2, y2, confidence))
    return detections

# Path to the detections.txt file
detection_file_path = '/home/kunj/Downloads/cv/MOT15/train/ADL-Rundle-6/det/det.txt'

# Path to the folder containing images
images_folder_path = '/home/kunj/Downloads/cv/MOT15/train/ADL-Rundle-6/img1'

# Initialize SORT tracker
tracker = Sort()

# Read detections
detections = read_detections(detection_file_path)

# Prepare to save the tracker's output
tracker_outputs = []

# Process each frame
for frame_number in sorted(detections.keys()):
    # Assuming image filenames are in the format: frame_number.jpg
    frame_path = f"{images_folder_path}/{int(frame_number):06}.jpg"
    frame = cv2.imread(frame_path)

    # Get detections for this frame, if any
    frame_detections = detections.get(frame_number, [])
    dets = np.array(frame_detections)

    # Update tracker with current frame detections
    trackers = tracker.update(dets)

    # Collect tracker outputs for saving
    for d in trackers:
        d = d.astype(np.int32)
        bbox = (d[0], d[1], d[2] - d[0], d[3] - d[1])  # Convert to x, y, width, height
        tracker_outputs.append((frame_number, d[4], *bbox))  # Include frame_number and tracker_id

        # Optionally draw tracking results on the frame
        cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), (255, 255, 255), 2)
        cv2.putText(frame, str(d[4]), (d[0], d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame with tracking results
    cv2.imshow('Frame', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# When everything is done, release the video capture object
cv2.destroyAllWindows()

# Save the tracker's output to a file in a format compatible with the ground truth
with open('tracker_output_corrected.txt', 'w') as f:
    for line in tracker_outputs:
        # Format: frame_number, tracker_id, x, y, w, h, 1, -1, -1, -1
        f.write(','.join(map(str, line)) + ",1,-1,-1,-1\n")
