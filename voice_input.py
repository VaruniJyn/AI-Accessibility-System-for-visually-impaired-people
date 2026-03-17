import speech_recognition as sr

def listen():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening for a command...")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Reduce noise
        try:
            audio = recognizer.listen(source, timeout=5)
            user_text = recognizer.recognize_google(audio)
            print(f"User said: {user_text}")
            return user_text.lower()
        except sr.WaitTimeoutError:
            print("Timeout: No speech detected.")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio.")
            return None

# Test speech recognition
if __name__ == "__main__":
    listen()
