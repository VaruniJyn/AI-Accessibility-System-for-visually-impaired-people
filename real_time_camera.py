import cv2
import time

def capture_image():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return None

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture image")
        return None

    image_path = "captured_image.jpg"
    cv2.imwrite(image_path, frame)

    cap.release()
    return image_path

# Test function
if __name__ == "__main__":
    img_path = capture_image()
    if img_path:
        print(f"Image saved at: {img_path}")
