import cv2

# Use the correct IP and port from DroidCam
stream_url = "http://192.168.1.4:4747/mjpegfeed"

# Open video stream
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Could not open video stream")
else:
    print("Video stream started successfully!")

# Read and display video
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    cv2.imshow("Mobile Camera Stream", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
