import os
import numpy as np


def get_true_embeds(embeds, targets):
    set_targets = set(targets)
    true_embeds = []

    for e in set_targets:
        ixs = np.where(targets == e)
        true_embeds.append(np.mean(embeds[ixs], axis=0))

    return np.array(true_embeds)


def get_data(data_dir):
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


def train_test_split(img_paths, test_size=0.2, shuffle=True):
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
