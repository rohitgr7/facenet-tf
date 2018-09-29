# TODO: Set the image size in args
# TODO: Set the video path correctly
# Add rectangles to the final window

import cv2
import argparse
import tensorflow as tf
import pickle

from utils import *


def _load_classifier_model(model_path):
    pass


def _detect(frame):
    pass


def _create_canvas(frame, preds, rects):
    pass


def _main(args):
    video_capture = cv2.VideoCapture(args.v_path)
    window_name = 'Video'

    with tf.Graph().as_default():
        with tf.Session() as sess:
            input_placeholder, embeddings_tensor, phase_train_placholder = get_model_tensors(args.model_path)

            classifier_model = _load_classifier_model(args.classifier_path)

            while True:
                _, frame = video_capture.read()

                images, rects = detect_and_align([frame], 160, meta_dir, training=False)
                images, rects = np.squeeze(images), np.squeeze(rects)

                feed_dict = {input_placeholder: images, phase_train_tensor: False}
                embeds = sess.run(embeddings, feed_dict=feed_dict)
                preds = _detect(embeds, classifier_model)

                canvas = _create_canvas(frame, preds, rects)
                cv2.imshow(window_name, canvas)

                if cv2.waitkey(10) & 0xFF == ord('q'):
                    break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser for detection')

    parser.add_argument('--v_path', type=str, default='./sample.mp4')
    parser.add_arguement('--model_path', type=str, default='./models/embed_model/20170512-110547/20170512-110547.pb')
    parser.add_arguement('--classifier_path', type=str, default='./models/classifier/model.pkl')

    args = parser.parse_args()
    _main(args)
