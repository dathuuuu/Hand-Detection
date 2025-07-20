import cv2
import numpy as np

# Capture video frames from the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Apply thresholding to detect skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the contour with the maximum area
    max_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to simplify it
    approx = cv2.approxPolyDP(max_contour, 0.02 * cv2.arcLength(max_contour, True), True)
    
    # Recognize the gesture based on the approximated contour
    if len(approx) == 5:
        # Recognize the "open hand" gesture
        cv2.putText(frame, "Open Hand", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    elif len(approx) == 3:
        # Recognize the "closed hand" gesture
        cv2.putText(frame, "Closed Hand", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    elif len(approx) == 4:
        # Recognize the "good luck" gesture
        cv2.putText(frame, "Good Luck", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    elif len(approx) == 2:
        # Recognize the "okay" gesture
        cv2.putText(frame, "Okay", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    elif len(approx) == 6:
        # Recognize the "number 0" gesture
        cv2.putText(frame, "0", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    elif len(approx) == 7:
        # Recognize the "number 1" gesture
        cv2.putText(frame, "1", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    elif len(approx) == 8:
        # Recognize the "number 2" gesture
        cv2.putText(frame, "2", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    elif len(approx) == 9:
        # Recognize the "number 3" gesture
        cv2.putText(frame, "3", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    elif len(approx) == 10:
        # Recognize the "number 4" gesture
        cv2.putText(frame, "4", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    elif len(approx) == 11:
        # Recognize the "number 5" gesture
        cv2.putText(frame, "5", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Display the output
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    
    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

