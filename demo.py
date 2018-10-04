import cv2
import argparse
import tensorflow as tf
import pickle
import numpy as np

from utils.utils import detect_and_align, get_model_tensors, get_face_detection_models
from utils.test_utils import detect, create_canvas


def _video_verification(args):
    video_capture = cv2.VideoCapture(args.f_path)
    window_name = 'Video'

    with tf.Graph().as_default():
        with tf.Session() as sess:
            input_placeholder, embeddings_tensor, phase_train_placeholder = get_model_tensors(args.model_path)

            with open(args.classifier_path, 'rb') as f:
                classifier_model, classes, true_embeds = pickle.load(f)

            face_detector, face_aligner = get_face_detection_models(args.meta_dir)

            while True:
                _, frame = video_capture.read()

                images, rects = detect_and_align([frame], 160, face_detector, face_aligner, training=False)
                images, rects = np.squeeze(images), np.squeeze(rects)

                if len(images.shape) == 3:
                    images = images[np.newaxis, ...]
                    rects = rects[np.newaxis, ...]

                if len(images.shape) == 4:

                    feed_dict = {input_placeholder: images, phase_train_placeholder: False}
                    embeds = sess.run(embeddings_tensor, feed_dict=feed_dict)
                    probs, names, frects = detect(embeds, rects, classifier_model, classes, true_embeds, args.th)

                    if probs is not None:
                        frame = create_canvas(frame, probs, names, frects)

                cv2.imshow(window_name, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            video_capture.release()
            cv2.destroyAllWindows()


def _image_verification(args):
    window_name = 'Image'

    with tf.Graph().as_default():
        with tf.Session() as sess:
            input_placeholder, embeddings_tensor, phase_train_placeholder = get_model_tensors(args.model_path)

            with open(args.classifier_path, 'rb') as f:
                classifier_model, classes, true_embeds = pickle.load(f)

            face_detector, face_aligner = get_face_detection_models(args.meta_dir)

            frame = cv2.imread(args.f_path, cv2.COLOR_BGR2RGB)
            images, rects = detect_and_align([frame], 160, face_detector, face_aligner, training=False)
            images, rects = np.squeeze(images), np.squeeze(rects)

            if len(images.shape) == 3:
                images = images[np.newaxis, ...]
                rects = rects[np.newaxis, ...]

            if len(images.shape) == 4:

                feed_dict = {input_placeholder: images, phase_train_placeholder: False}
                embeds = sess.run(embeddings_tensor, feed_dict=feed_dict)
                probs, names, frects = detect(embeds, rects, classifier_model, classes, true_embeds, args.th)

                if probs is not None:
                    frame = create_canvas(frame, probs, names, frects)

            cv2.imshow(window_name, frame)
            cv2.waitKey(10000)
            cv2.destroyAllWindows()


def _main(args):

    if args.f_path.endswith('.mp4'):
        _video_verification(args)
    else:
        _image_verification(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser for detection')

    parser.add_argument('--f_path', type=str, required=True, help='Sample video/image path')
    parser.add_argument('--model_path', type=str, default='./models/embed_model/20170512-110547/20170512-110547.pb', help='Embed model path')
    parser.add_argument('--classifier_path', type=str, default='./models/classifier/model.pkl', help='Classifier path')
    parser.add_argument('--meta_dir', type=str, default='./meta', help='Meta directory')
    parser.add_argument('--th', type=float, default=0.6, help='Threshold to eliminate unverified faces')

    args = parser.parse_args()
    _main(args)
