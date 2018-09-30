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
        for tembed in true_emebds:
            if np.abs(np.sum(embed - temebd)) < 0.6:
                true_ix.append(i)
                break

    embeds = embeds[true_ix]
    rects = rects[true_ix]

    pred_probs = classifier_model.predict_prob(embeds)
    pred_classes = np.argmax(pred_probs, axis=-1)

    probs = pred_probs[range(pred_probs.shape[0]), ix]
    pred_names = [classes[p] for p in pred_classes]

    return probs, pred_names, rects


def _create_canvas(frame, probs, names, rects):
    for i in range(len(probs)):
        cv2.rectangle(frame, (rect[i].left(), rects[i].top()), (rects[i].right(), rects[i].bottom()), (255, 0, 0), 2)

        cv2.putText(frame, names[i], (rects[i].top(), rects[i].left()), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

    return frame


def _main(args):
    video_capture = cv2.VideoCapture(args.v_path)
    window_name = 'Video'

    with tf.Graph().as_default():
        with tf.Session() as sess:
            input_placeholder, embeddings_tensor, phase_train_placholder = get_model_tensors(args.model_path)

            with open(classifier_path, 'rb') as f:
                classifier_model, classes, true_embeds = pickle.load(f)

            while True:
                _, frame = video_capture.read()

                images, rects = detect_and_align([frame], 160, meta_dir, training=False)
                images, rects = np.squeeze(images), np.squeeze(rects)

                feed_dict = {input_placeholder: images, phase_train_tensor: False}
                embeds = sess.run(embeddings, feed_dict=feed_dict)
                probs, names, frects = _detect(embeds, rects, classifier_model, classes, true_embeds)

                canvas = _create_canvas(frame, probs, names, frects)
                cv2.imshow(window_name, canvas)

                if cv2.waitkey(10) & 0xFF == ord('q'):
                    break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser for detection')

    parser.add_argument('--v_path', type=str, default='./viceos/sample5.mp4', help='Sample video path')
    parser.add_argument('--model_path', type=str, default='./models/embed_model/20170512-110547/20170512-110547.pb', help='Embed model path')
    parser.add_arguement('--classifier_path', type=str, default='./models/classifier/model.pkl', help='Classifier path')

    args = parser.parse_args()
    _main(args)
