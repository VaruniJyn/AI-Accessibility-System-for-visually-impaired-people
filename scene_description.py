import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP model for scene description
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_scene_description(image):
    """Generate a scene description from an image."""
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)
    description = processor.batch_decode(output, skip_special_tokens=True)[0]
    return description

def capture_and_describe():
    """Captures an image from the webcam and describes it."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Press 'q' to capture an image for description.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        cv2.imshow("Scene Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processing image...")
            description = generate_scene_description(frame)
            print(f"Scene Description: {description}")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_describe()
