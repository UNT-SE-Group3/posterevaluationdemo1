!pip install gradio
!pip install opencv-python-headless
!pip install pytesseract
!sudo apt install tesseract-ocr
import gradio as gr
import cv2
import pytesseract
from pytesseract import Output
import numpy as np

CONFIDENCE_THRESHOLD = 60

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise Exception("Error loading image")
    return image

def extract_text(image):
   
    text = pytesseract.image_to_string(image, output_type=Output.STRING)
    return text.lower() 

def assess_common_words(text):
  
    common_words = ['introduction', 'objective', 'results', 'conclusion', 'further scope', 'acknowledgments', 'hypothesis', 'abstract', 'Abstract', 'methodology', 'discussion', 'references', 'literature review', 'data analysis', 'experiment', 'materials', 'method', 'figure', 'table', 'figure 1', 'table 1', 'discussion', 'appendix', 'author', 'presentation']
    unique_common_words = set(common_words)
    word_count = len([word for word in text.split() if word in unique_common_words])
    score = word_count * 3
    return score

def detect_title(image):
   
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

def detect_images_and_graphs(image):
   
    non_text_area = cv2.subtract(255, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    _, non_text_area = cv2.threshold(non_text_area, 128, 255, cv2.THRESH_BINARY)
    non_text_pixels = cv2.countNonZero(non_text_area)

    if non_text_pixels > 0:
        return 10
    else:
        return 0

def detect_logo(image):
 
    logo_region = image[:100, -100:]
    gray_logo = cv2.cvtColor(logo_region, cv2.COLOR_BGR2GRAY)
    _, binary_logo = cv2.threshold(gray_logo, 128, 255, cv2.THRESH_BINARY)
    non_zero_pixels = cv2.countNonZero(binary_logo)

    if non_zero_pixels > 0:
        return 10
    else:
        return 0
def predict(image):
   
    img_np = image

    text = extract_text(img_np)

    common_words_score = assess_common_words(text)
    title_score = detect_title(img_np)
    images_and_graphs_score = detect_images_and_graphs(img_np)
    logo_score = detect_logo(img_np)

    total_score = common_words_score + title_score + images_and_graphs_score + logo_score

    return {
        "Common Words Score": common_words_score,
        "Title Score": title_score,
        "Images and Graphs Score": images_and_graphs_score,
        "Logo Score": logo_score,
        "Total Score": total_score
    }


iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(shape=(None, None), image_mode='RGB', label="Upload Image"),  
    outputs=gr.outputs.JSON(label="Scores")
)

iface.launch()