import sys
from google.cloud import vision

import io

def detect_objects(image_path):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    return image

client = vision.ImageAnnotatorClient()
client.

for arg in sys.argv[1:]:
    print(detect_objects(arg))    
