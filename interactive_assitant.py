import speech_recognition as sr
import pyttsx3
import subprocess
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from object_detection import detect_objects  # Import object detection
from scene_description import describe_scene  # Import scene description

# Initialize speech recognition and text-to-speech
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

# Load a conversational model
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def speak(text):
    """Speaks the given text using pyttsx3."""
    if text.strip():  # Ensure there's text to speak
        print(f"Speaking: {text}")  # Debugging output
        tts_engine.say(text)
        tts_engine.runAndWait()

def listen():
    """Listens to the user and converts speech to text."""
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=2)  # Increase noise adjustment time
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)  # Extend timeout
            user_input = recognizer.recognize_google(audio)
            print(f"You said: {user_input}")
            return user_input.lower()
        except sr.WaitTimeoutError:
            print("Timeout: No speech detected.")
            speak("I didn't hear anything. Can you repeat?")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio.")
            speak("Sorry, I couldn't understand that.")
            return None
        except sr.RequestError:
            print("Speech recognition service error.")
            return None

def generate_response(user_text):
    """Generate an AI response using the conversational model."""
    inputs = tokenizer.encode(user_text + tokenizer.eos_token, return_tensors="pt")
    
    with torch.no_grad():
        output = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    
    response = tokenizer.decode(output[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response

def chat():
    while True:
        user_text = listen()
        if user_text:
            if "describe the room" in user_text.lower():
                detected_objects = detect_objects()  # Call object detection function
                if detected_objects:
                    response = f"I see {', '.join(detected_objects)} in the room."
                    print(f"AI: {response}")
                    speak(response)
                else:
                    speak("I could not detect any objects.")
            else:
                response = generate_response(user_text)
                print(f"AI: {response}")
                speak(response)

if __name__ == "__main__":
    chat()
