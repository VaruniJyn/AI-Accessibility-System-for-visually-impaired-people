from ocr_function import extract_text

# Path to the image you want to test
img_path = 'uploaded_images/cat.jpg'  # Replace with the actual image path

# Call the extract_text function
extracted_text = extract_text(img_path)

# Print the result
if extracted_text:
    print("Extracted Text:")
    print(extracted_text)
else:
    print("No text extracted.")
