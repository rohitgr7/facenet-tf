# TODO: Set the image size in args
# TODO: Set the video path correctly

import cv2
import argparse
import tensorflow as tf
import pickle
import numpy as np

from utils import *


def _detect(embeds, rects, classifier_model, classes, true_embeds):
    true_ix = []

    for i, embed in enumerate(embeds):
        if np.any(np.abs(np.sum(true_embeds - embed, axis=-1)) < 0.7):
            true_ix.append(i)

    if len(true_ix) == 0:
        return None, None, None

    embeds = embeds[true_ix]
    rects = rects[true_ix]

    pred_probs = classifier_model.predict_proba(embeds)
    pred_classes = np.argmax(pred_probs, axis=-1)

    probs = pred_probs[range(pred_probs.shape[0]), pred_classes]
    pred_names = [classes[p] for p in pred_classes]

    return probs, pred_names, rects


def _create_canvas(frame, probs, names, rects):
    for i in range(len(probs)):
        cv2.rectangle(frame, (rects[i].left(), rects[i].top()), (rects[i].right(), rects[i].bottom()), (255, 0, 0), 2)

        cv2.putText(frame, names[i], (rects[i].left(), rects[i].top()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame


def _main(args):
    video_capture = cv2.VideoCapture(args.v_path)
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
                    probs, names, frects = _detect(embeds, rects, classifier_model, classes, true_embeds)

                    if probs is not None:
                        frame = _create_canvas(frame, probs, names, frects)

                cv2.imshow(window_name, frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser for detection')

    parser.add_argument('--v_path', type=str, default='./videos/sample1.mp4', help='Sample video path')
    parser.add_argument('--model_path', type=str, default='./models/embed_model/20170512-110547/20170512-110547.pb', help='Embed model path')
    parser.add_argument('--classifier_path', type=str, default='./models/classifier/model.pkl', help='Classifier path')
    parser.add_argument('--meta_dir', type=str, default='./meta', help='Meta directory')

    args = parser.parse_args()
    _main(args)
