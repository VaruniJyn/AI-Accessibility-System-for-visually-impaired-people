import cv2
import os
import torch
import speech_recognition as sr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import pyttsx3
import numpy as np

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load Object Detection Model (YOLOv5)
object_model = YOLO("yolov5s.pt")

# Load Scene Description Model (BLIP)
scene_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
scene_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize Text-to-Speech (TTS) Engine
tts_engine = pyttsx3.init()

# Initialize Speech Recognition
recognizer = sr.Recognizer()

# Start Camera
cap = cv2.VideoCapture(0)

# Get frame dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

def speak(text):
    """Convert text to speech."""
    tts_engine.say(text)
    tts_engine.runAndWait()

def get_direction(x1, y1, x2, y2):
    """Determine object position in the frame (Left, Right, Center, Top, Bottom)."""
    obj_center_x = (x1 + x2) // 2
    obj_center_y = (y1 + y2) // 2

    horizontal_position = "center"
    vertical_position = "middle"

    if obj_center_x < frame_width * 0.3:
        horizontal_position = "left"
    elif obj_center_x > frame_width * 0.7:
        horizontal_position = "right"

    if obj_center_y < frame_height * 0.3:
        vertical_position = "top"
    elif obj_center_y > frame_height * 0.7:
        vertical_position = "bottom"

    return f"{vertical_position} {horizontal_position}"

def generate_scene_description(frame):
    """Generate real-time scene description using BLIP model."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = scene_processor(image, return_tensors="pt")  
    output = scene_model.generate(**inputs)
    description = scene_processor.batch_decode(output, skip_special_tokens=True)[0]
    return description

def listen_for_command():
    """Listen for voice commands from the user."""
    with sr.Microphone() as source:
        print("Listening for command...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)  
            command = recognizer.recognize_google(audio).lower()
            return command
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            return None
        except sr.WaitTimeoutError:
            return None

def find_safest_exit_path(objects):
    """Simple logic to find an obstacle-free path."""
    directions = []

    # If there's a clear way forward
    if not any("front" in obj for obj in objects):
        directions.append("Move forward.")

    # If obstacles exist, suggest alternative paths
    if any("left" in obj for obj in objects):
        directions.append("Avoid the left side.")
    if any("right" in obj for obj in objects):
        directions.append("Avoid the right side.")

    # If front is blocked, check sides
    if any("front" in obj for obj in objects):
        directions.append("Turn slightly left or right to avoid obstacles.")

    return " ".join(directions) if directions else "Clear path ahead."

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Object Detection
    results = object_model.predict(frame)
    detected_objects = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            position = get_direction(x1, y1, x2, y2)
            detected_objects.append(f"{label} at {position}")

            # Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({position})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display Camera Feed
    cv2.imshow("AI Accessibility System", frame)

    # Every 30 frames, check if user gives a command
    if frame_count % 30 == 0:
        command = listen_for_command()

        if command:
            print(f"User said: {command}")

            if "exit" in command or "guide me" in command:
                scene_description = generate_scene_description(frame)
                safe_route = find_safest_exit_path(detected_objects)
                final_instruction = f"{safe_route}. Scene: {scene_description}"
                print(final_instruction)
                speak(final_instruction)

    frame_count += 1

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
