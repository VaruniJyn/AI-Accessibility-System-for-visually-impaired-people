from ocr_function import extract_text

img_path = 'uploaded_images/cat.jpg'

# Run the full OCR pipeline
extracted_text = extract_text(img_path)

if extracted_text:
    print(f"Extracted Text: {extracted_text}")
else:
    print("OCR extraction failed.")
