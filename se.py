!pip install opencv-python-headless
!pip install pytesseract
!sudo apt install tesseract-ocr
import cv2
import pytesseract
from pytesseract import Output
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise Exception("Error loading image")
    return image
def extract_text(image):
    text = pytesseract.image_to_string(image, output_type=Output.STRING)
    return text.lower()