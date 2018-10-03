import cv2
import numpy as np


def detect(embeds, rects, classifier_model, classes, true_embeds, threshold=0.6):
    true_ix = []

    for i, embed in enumerate(embeds):
        if np.any(np.abs(np.sum(true_embeds - embed, axis=-1)) < threshold):
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


def create_canvas(frame, probs, names, rects):
    for i in range(len(probs)):
        cv2.rectangle(frame, (rects[i].left(), rects[i].top()), (rects[i].right(), rects[i].bottom()), (255, 0, 0), 2)

        cv2.putText(frame, names[i], (rects[i].left(), rects[i].top()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame
