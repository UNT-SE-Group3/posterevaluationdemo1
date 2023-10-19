!pip install opencv-python-headless
import cv2
def detect_logo(image):
    # Detect if there is a logo in the top or top-right corner of the image
    logo_region = image[:100, -100:]
    gray_logo = cv2.cvtColor(logo_region, cv2.COLOR_BGR2GRAY)
    _, binary_logo = cv2.threshold(gray_logo, 128, 255, cv2.THRESH_BINARY)
    non_zero_pixels = cv2.countNonZero(binary_logo)

    if non_zero_pixels > 0:
        return 10
    else:
        return 0