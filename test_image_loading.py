from PIL import Image

# Path to the image
img_path = 'uploaded_images/cat.jpg'

# Test loading the image
try:
    img = Image.open(img_path)
    img.show()  # Opens the image in the default image viewer
    print(f"Image loaded: {img_path}")
except Exception as e:
    print(f"Error loading image: {e}")
