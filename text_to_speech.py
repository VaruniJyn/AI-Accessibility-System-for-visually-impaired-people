import pyttsx3

def speak(text):
    """Convert text to speech."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Test AI speaking
if __name__ == "__main__":
    speak("Hello! I am your AI assistant.")
