
!pip install pytesseract
!sudo apt install tesseract-ocr
import pytesseract
from pytesseract import Output
def extract_text(image):
    text = pytesseract.image_to_string(image, output_type=Output.STRING)
    return text.lower()  

def assess_common_words(text):
    # Count the number of common words present in the text without duplicates
    common_words = ['introduction', 'objective', 'results', 'conclusion', 'further scope', 'acknowledgments', 'hypothesis', 'abstract', 'Abstract', 'methodology', 'discussion', 'references', 'literature review', 'data analysis', 'experiment', 'materials', 'method', 'figure', 'table', 'figure 1', 'table 1', 'discussion', 'appendix', 'author', 'presentation']
    unique_common_words = set(common_words)
    word_count = len([word for word in text.split() if word in unique_common_words])
    score = word_count * 3
    return score