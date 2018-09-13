import numpy as np
from PIL import Image
import dlib
import openface
import matplotlib.pyplot as plt


def load_image(image_path):
    return np.array(Image.open(image_path))


def detect_and_align(image_paths, image_size, training=True):
    img_faces = []

    for img_path in image_paths:
        img = load_image(img_path)

        predictor_model = './meta/shape_predictor_68_face_landmarks.dat'
        face_detector = dlib.get_frontal_face_detector()
        face_aligner = openface.AlignDlib(predictor_model)

        detected_faces = face_detector(img, 1)

        faces = []

        for i, face_rect in enumerate(detected_faces):

            aligned_face = face_aligner.align(image_size, img, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

            if training:
                faces.append(aligned_face)
                break
            else:
                faces.append((aligned_face, face_rect))

        img_faces.append(faces)

    return img_faces


def plot_image(img):
    plt.imshow(img / 255.)
