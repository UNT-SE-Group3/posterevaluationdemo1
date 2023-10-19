!pip install opencv-python-headless
import cv2
def detect_images_and_graphs(image):
    # Detect if there are any non-text areas that may contain images or graphs
    # For simplicity, this code detects non-white regions as potential images or graphs
    non_text_area = cv2.subtract(255, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    _, non_text_area = cv2.threshold(non_text_area, 128, 255, cv2.THRESH_BINARY)
    non_text_pixels = cv2.countNonZero(non_text_area)

    if non_text_pixels > 0:
        return 10
    else:
        return 0