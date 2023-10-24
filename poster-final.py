# !pip install gradio
# !pip install opencv-python-headless
# !pip install pytesseract
# !sudo apt install tesseract-ocr
# To install tesseract-ocr in windows machine download exe file from  https://github.com/UB-Mannheim/tesseract/wiki.
# To install tesseract in debian based linux distros follow the steps below
#yum-config-manager --add-repo https://download.opensuse.org/repositories/home:/Alexander_Pozdnyakov/CentOS_7/
#yum update
#yum install tesseract 

import gradio as gr
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import json
import multiprocessing

# Constants and Configurations
CONFIDENCE_THRESHOLD = 60

def is_valid_image(image):
    """
    Check if the image is valid based on size and format.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.

    Returns:
        bool: True if the image is valid; False otherwise.
    Raises:
        Exception: if the provided image is not in BGR format then exception will be raised.
    """

    # Check the image dimensions
    min_height = 100
    min_width = 100

    max_height = 10000
    max_width = 10000

    if image.shape[0] < min_height or image.shape[1] < min_width:
        return False
    if image.shape[0] > max_height or image.shape[1] > max_width:
        return False
    if image.shape[-1] != 3 or image.shape[-2] != 3:
        raise Exception("Image format is not BGR.")

    return True

# Function to load an image from a file
def load_image(image_path):
    """
    Load an image from a file.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The loaded image as a NumPy array.
    Raises:
        Exception: If there is an error loading the image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise Exception("Error loading image")
    if not is_valid_image(image):
        raise Exception("Not a valid image.")
    return image

def extract_text(image):
    """
    Extract text from an image using pytesseract.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.

    Returns:
        str: The extracted text in lowercase for easier comparison.
    """
    # Extract text from the image using pytesseract
    text = pytesseract.image_to_string(image, output_type=Output.STRING)
    return text.lower()  # Convert text to lowercase for easier comparison

def assess_common_words(text, common_words):
    
    # Count the number of common words present in the text without duplicates
    """
    Count the number of common words present in the text without duplicates.

    Args:
        text (str): The text to assess.
        common_words (list): List of common words to check for in the text.

    Returns:
        int: A score based on the number of common words found.
    """
    unique_common_words = set(common_words)
    word_count = len([word for word in text.split() if word in unique_common_words])
    score = word_count * 3
    return score

def detect_title(image):
    # Detect if there is a title in the top middle of the image
    """
    Detect if there is a title in the top middle of the image.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.

    Returns:
        int: A title detection score.
    """
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
    # Detect if there are any non-text areas that may contain images or graphs
    # For simplicity, this code detects non-white regions as potential images or graphs
    """
    Detect if there are any non-text areas that may contain images or graphs.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.

    Returns:
        int: A score indicating the presence of images or graphs.
    """
    non_text_area = cv2.subtract(255, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    _, non_text_area = cv2.threshold(non_text_area, 128, 255, cv2.THRESH_BINARY)
    non_text_pixels = cv2.countNonZero(non_text_area)

    if non_text_pixels > 0:
        return 10
    else:
        return 0

def detect_logo(image):
    """
    Detect if there is a logo in the top or top-right corner of the image.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.

    Returns:
        int: A logo detection score.
    """
    # Detect if there is a logo in the top or top-right corner of the image
    logo_region = image[:100, -100:]
    gray_logo = cv2.cvtColor(logo_region, cv2.COLOR_BGR2GRAY)
    _, binary_logo = cv2.threshold(gray_logo, 128, 255, cv2.THRESH_BINARY)
    non_zero_pixels = cv2.countNonZero(binary_logo)

    if non_zero_pixels > 0:
        return 10
    else:
        return 0
def predict(image, common_words):
    """
    Predict assessment scores based on image analysis.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.
        common_words (list): List of common words to check for in the text.

    Returns:
        dict: A dictionary containing assessment scores.
    """
    # Process the image directly without reading from file
    # Since we're using 'numpy' type for the Gradio input, the 'image' is now a numpy array
    img_np = image

    text = extract_text(img_np)

    common_words_score = assess_common_words(text, common_words)
    title_score = detect_title(img_np)
    images_and_graphs_score = detect_images_and_graphs(img_np)
    logo_score = detect_logo(img_np)

    total_score = common_words_score + title_score + images_and_graphs_score + logo_score
    ret_dict = {
        "Common Words Score": common_words_score,
        "Title Score": title_score,
        "Images and Graphs Score": images_and_graphs_score,
        "Logo Score": logo_score,
        "Total Score": total_score
    }
    print(ret_dict)
    return ret_dict


if __name__ == "__main__":
    image = load_image("sample.jpeg")
    all_common_words = []
    with open("config.json", 'r') as json_file:
        data = json.load(json_file)
        for lang, words in data.items():
            all_common_words.extend(words)

    process = multiprocessing.Process(target=predict, args=(image, all_common_words,))
    process.start()
    process.join()

# # Create the Gradio interface
# iface = gr.Interface(
#     fn=predict,
#     inputs=gr.Image(shape=(None, None), image_mode='RGB', label="Upload Image"),  # Adjusted input parameters
#     outputs=gr.outputs.JSON(label="Scores")
# )

# # Run the interface
# iface.launch()