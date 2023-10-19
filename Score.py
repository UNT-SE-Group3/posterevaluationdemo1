!pip install gradio
!pip install opencv-python-headless
!pip install pytesseract
!sudo apt install tesseract-ocr
import gradio as gr
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
def predict(image):
    # Process the image directly without reading from file
    # Since we're using 'numpy' type for the Gradio input, the 'image' is now a numpy array
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

# Create the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(shape=(None, None), image_mode='RGB', label="Upload Image"),  # Adjusted input parameters
    outputs=gr.outputs.JSON(label="Scores")
)

# Run the interface
iface.launch()