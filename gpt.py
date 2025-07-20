import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Ensure the path is correct

def identify_gesture(fingers_up):
    if fingers_up == 1:
        return "OK"  # Assuming one finger is extended for "OK"
    elif fingers_up == 2:
        return "Good Luck"  # Assuming two fingers are extended for "Good Luck"
    return "Unknown"

def count_fingers(detections):
    # Basic logic to count fingers based on detection
    # You would replace this logic with more specific finger detection
    fingers_up = 0
    if len(detections) > 0:
        # If a hand is detected, we assume it's either one or two fingers up
        # You may replace this with more precise finger analysis later
        fingers_up = len(detections)  # Simplified
    return fingers_up

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Extract results
    hands_detected = []

    for result in results:
        for detection in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = detection
            if int(cls) == 0:  # Assuming '0' is the class ID for hands
                hands_detected.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Count fingers (this is a simplification; adjust as needed)
    fingers_up = count_fingers(hands_detected)

    # Identify gesture based on counted fingers
    gesture = identify_gesture(fingers_up)

    # Display the gesture on the frame
    cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Show the frame
    cv2.imshow('Hand Gesture Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
