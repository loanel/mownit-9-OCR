import pytesseract
from PIL import Image
import numpy as np

def main():
    img = "test_longer.jpg"
    print("\nAttempting to retrieve text from scanned picture, this might take a while for bigger documents")
    text = pytesseract.image_to_string(Image.open(img))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    print(text)

main()