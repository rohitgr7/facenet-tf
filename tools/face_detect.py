import argparse
import dlib
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, help='Image file path')
args = parser.parse_args()

face_detector = dlib.get_frontal_face_detector()

win = dlib.image_window()

# Load the image
img = np.array(Image.open(args.f))

# Run the HOG face detector on the image data
detected_faces = face_detector(img, 1)

print(f'Found {len(detected_faces)} faces in the image file {args.f}')

# Show the desktop window with the image
win.set_image(img)

# Loop through each face we found in the image
for i, face_rect in enumerate(detected_faces):

    # Detected faces are returned as an object with the coordinates
    # of the top, left, right and bottom edges
    print(f'- Face #{i} found at Left: {face_rect.left()} Top: {face_rect.top()} Right: {face_rect.right()} Bottom: {face_rect.bottom()}')

    # Draw a box around each face we found
    win.add_overlay(face_rect)

dlib.hit_enter_to_continue()
