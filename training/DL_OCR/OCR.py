import pytesseract
import cv2
import argparse
import os
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\seanq\AppData\Local\Tesseract-OCR\tesseract.exe'

# two arguments : "image" for giving path of image
#  "preprocess" for methods of pre-process : thresh(default) and blur
ap = argparse.ArgumentParser(description='Import and use ocr')
ap.add_argument("-i", "--image", required=True)

ap.add_argument("-p", "--preprocess", type=str, default="thresh")

args = vars(ap.parse_args())

# read the image and convert it into gray
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#thresh or blur
# check if it needs to preprocess or not
if args["preprocess"] == "thresh":
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# check if it needs to delete median blurring to remove noise or not
elif args["preprocess"] == "blur":
    gray = cv2.medianBlur(gray, 3)

# write the grayscale image as a temporary file
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

# load the image as a PIL/Pillow image, apply OCR, and then delete the temporary file
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)

# show the output images
cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
