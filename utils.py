import numpy as np
from PIL import Image
import dlib
import openface
import matplotlib.pyplot as plt
import tensorflow as tf
import os


def load_image(image_path):
    return np.array(Image.open(image_path))


def detect_and_align(images, image_size, meta_dir, training=True):
    img_faces = []
    img_rects = []

    for image in images:
        if os.path.isfile(image):
            img = load_image(image)
        else:
            img = image

        predictor_model = os.path.join(meta_dir, 'shape_predictor_68_face_landmarks.dat')
        face_detector = dlib.get_frontal_face_detector()
        face_aligner = openface.AlignDlib(predictor_model)

        detected_faces = face_detector(img, 1)

        faces = []
        rects = []

        for face_rect in detected_faces:

            aligned_face = face_aligner.align(image_size, img, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

            faces.append(aligned_face)

            if training:
                break

            rects.append(face_rect)

        img_faces.append(faces)
        img_rects.append(rects)

    return img_faces, img_rects


def plot_image(img):
    plt.imshow(img / 255.)


def _load_model(model_path, input_map=None):
    if os.path.isfile(model_path):
        with tf.gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def, input_map)

    else:
        meta_file, ckpt_file = get_model_filenames(model_path)
        saver = tf.train.import_meta_graph(os.path.join(model_path, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_path, meta_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_file = [f for f in files if f.endswith('.meta')][0]
    ckpt = tf.train.get_checkpoint_state(model_dir)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file


def get_model_tensors(model_path):
    _load_model(model_path)
    print('Embedding model loaded')

    input_placeholder = tf.get_default_graph().get_tensor_by_name('import/embeddings:0')
    embeddings_tensor = tf.get_default_graph().get_tensor_by_name('import/embeddings:0')
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('import/phase_train:0')

    return input_placeholder, embeddings_tensor, phase_train_placeholder
