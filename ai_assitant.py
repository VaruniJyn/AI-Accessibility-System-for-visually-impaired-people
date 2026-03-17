import torch
import cv2
import pyttsx3
import speech_recognition as sr
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGenerationln
from PIL import Image

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load object detection model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load BLIP model for scene description
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def speak(text):
    """Converts text to speech."""
    engine.say(text)
    engine.runAndWait()


def listen():
    """Captures audio and converts to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("AI Speaking: I am listening...")
        speak("I am listening...")
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio)
            print(f"You said: {command}")
            return command.lower()
        except sr.WaitTimeoutError:
            print("AI Speaking: No speech detected.")
            speak("I didn't hear anything. Can you repeat?")
        except sr.UnknownValueError:
            print("AI Speaking: Sorry, I couldn't understand that.")
            speak("Sorry, I couldn't understand that.")
        return None


def detect_objects_real_time():
    """Performs real-time object detection and provides direction feedback."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Error: Could not open camera.")
        return

    speak("Starting real-time object detection. Move carefully.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.pandas().xyxy[0]

        if len(detections) > 0:
            object_info = []
            width = frame.shape[1]

            for _, row in detections.iterrows():
                name = row["name"]
                x_center = (row["xmin"] + row["xmax"]) / 2

                # Determine object's location
                if x_center < width * 0.3:
                    position = "left"
                elif x_center > width * 0.7:
                    position = "right"
                else:
                    position = "center"

                object_info.append(f"{name} on the {position}")

            detected_objects = ", ".join(set(object_info))
            print(f"Detected: {detected_objects}")
            speak(f"I see: {detected_objects}")

        cv2.imshow("Real-Time Object Detection", results.render()[0])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def assist_navigation():
    """Guides user in real-time to avoid obstacles and move safely."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Error: Could not open camera.")
        return

    speak("Navigation mode activated. Move slowly, I will guide you.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.pandas().xyxy[0]

        if len(detections) > 0:
            width = frame.shape[1]
            directions = {"left": 0, "right": 0, "center": 0}

            for _, row in detections.iterrows():
                x_center = (row["xmin"] + row["xmax"]) / 2
                if x_center < width * 0.3:
                    directions["left"] += 1
                elif x_center > width * 0.7:
                    directions["right"] += 1
                else:
                    directions["center"] += 1

            # Provide movement guidance based on object positions
            if directions["center"] > 0:
                speak("Obstacle ahead. Turn left or right.")
            elif directions["left"] > 0:
                speak("Obstacle on your left. Move slightly to the right.")
            elif directions["right"] > 0:
                speak("Obstacle on your right. Move slightly to the left.")
            else:
                speak("Path is clear. Move forward.")

        cv2.imshow("Navigation Assistance", results.render()[0])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def describe_scene_real_time():
    """Provides real-time scene description."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Error: Could not open camera.")
        return

    speak("Real-time scene description activated. Move the camera slowly.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(image, return_tensors="pt")
        output = blip_model.generate(**inputs)
        description = processor.decode(output[0], skip_special_tokens=True)

        print(f"Scene Description: {description}")
        speak(f"I see: {description}")

        cv2.imshow("Real-Time Scene Description", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main AI assistant loop."""
    speak("Hello! I am your AI assistant. What would you like to do?")
    
    while True:
        command = listen()
        if command is None:
            continue

        # Check for keywords in the command
        if "exit" in command or "quit" in command:
            speak("Goodbye!")
            break
        elif "detect" in command or "object" in command:
            speak("Starting real-time object detection.")
            detect_objects_real_time()
        elif "scene" in command or "describe" in command or "camera" in command:
            speak("Starting real-time scene description.")
            describe_scene_real_time()
        elif "help me walk" in command or "navigation" in command or "avoid obstacle" in command:
            speak("Helping you navigate safely.")
            assist_navigation()
        else:
            speak("Sorry, I didn't understand. Try saying 'detect objects' or 'help me walk'.")


if __name__ == "__main__":
    main()
