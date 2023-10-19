#sample python file
!pip install opencv-python-headless
import cv2
def detect_title(image):
    # Detect if there is a title in the top middle of the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_contour_area = 0
    title_score = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 0.2 * image.shape[0] and w > 0.2 * image.shape[1]:
            contour_area = cv2.contourArea(contour)
            if contour_area > max_contour_area:
                max_contour_area = contour_area
                title_score = 10

    return title_score