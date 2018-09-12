from PIL import Image
import numpy as np
import dlib
import openface
import argparse
import os


def _main(args):
    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()
    face_aligner = openface.AlignDlib(args.lmarks)

    # Load the image
    img = np.array(Image.open(args.f))

    # Run the HOG face detector on the image data
    detected_faces = face_detector(img, 1)

    print(f'Found {len(detected_faces)} faces in the image file {args.f}')

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):

        # Detected faces are returned as an object with the coordinates
        # of the top, left, right and bottom edges
        print(f'- Face #{i} found at Left: {face_rect.left()} Top: {face_rect.top()} Right: {face_rect.right()} Bottom: {face_rect.bottom()}')

        # Use openface to calculate and perform the face alignment
        aligned_face = face_aligner.align(args.img_size, img, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        Image.fromarray(aligned_face.astype(np.uint8)).save(f'{args.out_dir}/image-{i}.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', type=str, help='Image file path')
    parser.add_argument('--lmarks', type=str, default='../meta/shape_predictor_68_face_landmarks.dat', help='Landmarks file')
    parser.add_argument('--out_dir', type=str, default='./align', help='Directory to store aligned faces')
    parser.add_argument('--img_size', type=int, default=224, help='Final size of the aligned aimages')

    args = parser.parse_args()
    _main(args)
