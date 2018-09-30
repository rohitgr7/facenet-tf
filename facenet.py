import tensorflow as tf
import numpy as np
import os
import math
import argparse
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from utils import *


def _get_true_embeds(embeds, targets):
    set_targets = set(targets)
    true_embeds = []

    for e in set_targets:
        ixs = np.where(targets == e)
        true_embeds.append(np.mean(embeds[ixs], axis=0))

    return np.array(true_embeds)


def _get_data(data_dir):
    img_paths = []
    ix2names = {}
    dirs = os.listdir(data_dir)

    for i, name_dir in enumerate(dirs):
        dir_path = os.path.join(data_dir, name_dir)
        image_files = os.listdir(dir_path)

        img_file_paths = [os.path.join(dir_path, img_file) for img_file in image_files]

        img_paths.append(img_file_paths)
        ix2names[i] = name_dir

    return img_paths, ix2names


def _train_test_split(img_paths, test_size=0.2, shuffle=True):
    train_paths, test_paths = [], []
    train_targets, test_targets = [], []

    test_len = int(len(img_paths[0]) * test_size)
    train_len = len(img_paths[0][test_len:])

    for i, images in enumerate(img_paths):
        test_paths.extend(images[:test_len])
        test_targets.extend([i] * test_len)

        train_paths.extend(images[test_len:])
        train_targets.extend([i] * train_len)

    train_paths, test_paths = np.array(train_paths, dtype=np.object), np.array(test_paths, dtype=np.object)
    train_targets, test_targets = np.array(train_targets, dtype=np.int), np.array(test_targets, dtype=np.int)

    if shuffle:
        train_shuff = np.random.permutation(train_targets.shape[0])
        test_shuff = np.random.permutation(test_targets.shape[0])

        train_paths, train_targets = train_paths[train_shuff], train_targets[train_shuff]
        test_paths, test_targets = test_paths[test_shuff], test_targets[test_shuff]

    return train_paths, test_paths, train_targets, test_targets


def _main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:

            input_placeholder, embeddings_tensor, phase_train_placeholder = get_model_tensors(args.model_path)

            img_paths, ix2names = _get_data(args.data_dir)
            train_paths, test_paths, train_targets, test_targets = _train_test_split(img_paths)
            print('Train and test image paths loaded')

            num_imgs = train_paths.shape[0]
            embed_size = embeddings_tensor.get_shape()[1]
            train_embeddings = np.zeros((num_imgs, embed_size))
            num_batches = math.ceil(num_imgs / args.batch_size)

            face_detector, face_aligner = get_face_detection_models(args.meta_dir)

            for i in range(num_batches):
                st_ix = i * args.batch_size
                end_ix = min(st_ix + args.batch_size, num_imgs)
                img_batch_paths = train_paths[st_ix:end_ix]
                batch_imgs, _ = detect_and_align(img_batch_paths, args.img_size, face_detector, face_aligner)
                batch_imgs = np.squeeze(batch_imgs)

                if len(batch_imgs.shape) == 3:
                    batch_imgs = batch_imgs[np.newaxis, ...]

                train_feed_dict = {input_placeholder: batch_imgs, phase_train_placeholder: False}
                train_embeddings[st_ix:end_ix] = sess.run(embeddings_tensor, feed_dict=train_feed_dict)

            print('Embeddings Created')

            true_embeds = _get_true_embeds(train_embeddings, train_targets)

            classifier = SVC(kernel='linear', probability=True)
            classifier.fit(train_embeddings, train_targets)

            train_preds = classifier.predict(train_embeddings)
            print(f'Train accuracy score: {accuracy_score(train_targets, train_preds)}')

            with open(args.classifier_path, 'wb') as f:
                pickle.dump((classifier, ix2names, true_embeds), f)

            test_imgs, _ = detect_and_align(test_paths, args.img_size, face_detector, face_aligner)
            test_imgs = np.squeeze(test_imgs)
            test_feed_dict = {input_placeholder: test_imgs, phase_train_placeholder: False}
            test_embeddings = sess.run(embeddings_tensor, feed_dict=test_feed_dict)

            test_preds = classifier.predict(test_embeddings)
            print(f'Test accuracy score: {accuracy_score(test_targets, test_preds)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser for training classifier')

    parser.add_argument('--model_path', type=str, default='./models/embed_model/20170512-110547/20170512-110547.pb', help='Pretrained model to get embeddings')
    parser.add_argument('--classifier_path', type=str, default='./models/classifier/model.pkl', help='Path to save the classifier')
    parser.add_argument('--data_dir', type=str, default='./dataset/fm-2', help='Directory of the training data')
    parser.add_argument('--meta_dir', type=str, default='./meta', help='Meta directory')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=160, help='Initial image size for training images')

    args = parser.parse_args()

    _main(args)
