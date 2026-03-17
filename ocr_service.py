import pytesseract
import cv2

def extract_text(frame):
    # Convert frame to grayscale for better OCR results
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply OCR
    text = pytesseract.image_to_string(gray)

    return text
