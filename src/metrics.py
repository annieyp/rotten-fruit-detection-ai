import keras
import numpy as np


def iou(box, boxes):
    """IoU of one [ymin, xmin, ymax, xmax] box against an (N, 4) array."""
    ymin = np.maximum(box[0], boxes[:, 0])
    xmin = np.maximum(box[1], boxes[:, 1])
    ymax = np.minimum(box[2], boxes[:, 2])
    xmax = np.minimum(box[3], boxes[:, 3])

    inter = np.clip(ymax - ymin, 0, None) * np.clip(xmax - xmin, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + areas - inter
    return np.where(union > 0, inter / union, 0.0)


class DetectionMetrics(keras.callbacks.Callback):
    """Precision / recall / F1 over the validation set at IoU >= threshold.

    Detection has no single 'accuracy'; these are the standard per-epoch signals.
    A prediction is a true positive if it matches an unused ground-truth box of the
    same class with IoU >= iou_threshold.
    """

    def __init__(self, val_ds, iou_threshold=0.5, score_threshold=0.3, max_batches=None):
        super().__init__()
        self.val_ds = val_ds
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_batches = max_batches

    @staticmethod
    def _key(pred, *names):
        for name in names:
            if name in pred:
                return np.asarray(pred[name])
        raise KeyError(f"None of {names} found in prediction keys {list(pred)}")

    def on_epoch_end(self, epoch, logs=None):
        tp = fp = fn = 0

        dataset = self.val_ds.take(self.max_batches) if self.max_batches else self.val_ds
        for images, y in dataset:
            pred = self.model.predict(images, verbose=0)
            pred_boxes = self._key(pred, "boxes")
            pred_labels = self._key(pred, "labels", "classes")
            pred_scores = self._key(pred, "confidence", "scores")
            gt_boxes = np.asarray(y["boxes"])
            gt_labels = np.asarray(y["labels"])

            for i in range(pred_boxes.shape[0]):
                keep_gt = gt_labels[i] >= 0
                g_boxes = gt_boxes[i][keep_gt]
                g_labels = gt_labels[i][keep_gt]

                keep_pred = (pred_labels[i] >= 0) & (pred_scores[i] >= self.score_threshold)
                p_boxes = pred_boxes[i][keep_pred]
                p_labels = pred_labels[i][keep_pred]
                order = np.argsort(-pred_scores[i][keep_pred])
                p_boxes, p_labels = p_boxes[order], p_labels[order]

                matched = np.zeros(len(g_boxes), dtype=bool)
                for j in range(len(p_boxes)):
                    if len(g_boxes) == 0:
                        fp += 1
                        continue
                    ious = iou(p_boxes[j], g_boxes)
                    ious = np.where((g_labels == p_labels[j]) & ~matched, ious, 0.0)
                    best = int(np.argmax(ious))
                    if ious[best] >= self.iou_threshold:
                        tp += 1
                        matched[best] = True
                    else:
                        fp += 1
                fn += int((~matched).sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        logs = logs if logs is not None else {}
        logs["val_precision"] = precision
        logs["val_recall"] = recall
        logs["val_f1"] = f1
        print(f"  val_precision: {precision:.3f} - val_recall: {recall:.3f} - val_f1: {f1:.3f}")
