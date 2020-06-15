import numpy as np


def iou_score(gt, pred) -> float:
    '''Computes the iou score between ground truth box and predicted box
    Args:
    gt: 1d array ground truth box [xmin, ymin, xmax, ymax]
    pred:1d array predicted box [xmin, ymin, xmax, ymax]
    Ouput:
    iou_score: float [0,1]'''
    #  if xmin_pred >= xmax_gt or xmax_pred <= xmin_gt return 0 (no intersection along x_axis)
    #  the same for y axis
    if (pred[0] >= gt[2] or pred[2] <= gt[0]) or (pred[1] >= gt[3] or pred[3] <= gt[1]):
        return 0

    max_x = min(pred[2], gt[2])
    min_x = max(pred[0], gt[0])
    max_y = min(pred[3], gt[3])
    min_y = max(pred[1], gt[1])
    intersection = (max_x - min_x) * (max_y - min_y)
    pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])
    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
    iou_score = intersection / (pred_area + gt_area - intersection)
    return iou_score


def find_best_match(pred, gts):
    '''Given a bounding box predictions, return the best matched groud truth
    bounding box.
    Args:
    pred: np.array (4,), predicted box
    gts: np.array(N, 4), ground truth bounding boxes
    Output:
    (gt box, score): tuple (matched box, score of match)'''
    index, max_match = max(enumerate(gts), key=lambda box: iou_score(box[1], pred))
    # print(iou_score(max_match, pred))
    return index, max_match, iou_score(max_match, pred)


def map_score(gts, preds, thresholds: list) -> (float, list[float]):

    '''
    TODO: adapt the code for jit compiler
    Compute map between predictions and ground truth
         for threshold in thresholds:
            For box in predictions:
            find best matched score from ground truth (box, score)
            if score > threshold:
                tp ++
                update ground truch boxes (remove matched boxes)
            if score < threshold:
                fp ++
            for box in non-matched boxes:
                fn ++
            scores.add(tp/(tp+fp+fn))
        return the mean
    '''
    if len(gts) == 0:
        return 0

    thresholds = np.array(thresholds)
    positives = np.zeros((len(preds), len(thresholds)))
    matched_boxes = np.zeros((len(gts), len(thresholds)))

    for i, box in enumerate(preds):

        index, best_box, score = find_best_match(box, gts)
        mask = score >= thresholds
        matched_boxes[index, mask] += 1
        positives[i, mask] += 1

    tp = np.sum(matched_boxes>=1, axis=0)
    fp_matched = np.sum(matched_boxes, axis=0) - tp
    # print(fp_matched)
    fp_detected = len(preds) - np.sum(positives, axis=0)
    # print(fp_detected)
    fp = fp_matched + fp_detected
    fn = len(gts) - tp
    scores = tp /(tp + fp + fn)
    score = np.mean(scores)
    # print(fp)
    # print(positives)
    # print(matched_boxes)
    return score, scores
