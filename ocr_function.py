import pytesseract
import cv2


# Specify the path to Tesseract OCR executable (update if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(img_path):
    try:
        # Step 1: Load the image using OpenCV
        img = cv2.imread(img_path)

        # Step 2: Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 3: Apply thresholding to preprocess the image for OCR
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Step 4: Use Tesseract to extract text from the preprocessed image
        text = pytesseract.image_to_string(thresh)

        return text

    except Exception as e:
        print(f"Error in extract_text function: {e}")
        return None
