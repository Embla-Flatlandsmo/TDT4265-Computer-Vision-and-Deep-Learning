import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes

def calculate_box_area(box):
    """Calculate the area of a box"
    
    Args:
        box (np.array of floats): box corners
            [xmin, ymin, xmax, ymax] 

        returns:
            float: area of the box
    """
    return (box[2]-box[0])*(box[3]-box[1])


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # Task 2b
    # Compute intersection

    intersection_box = [
        max(prediction_box[0], gt_box[0]), #xmin
        max(prediction_box[1], gt_box[1]), #ymin
        min(prediction_box[2], gt_box[2]), #xmax
        min(prediction_box[3], gt_box[3])  #ymax
    ]
    if intersection_box[0] > intersection_box[2] or intersection_box[1] > intersection_box[3]:
        return 0
    intersection_area = calculate_box_area(intersection_box)
    # if (intersection_area < 0):
    #     return 0

    # Compute union
    # A ∪ B = A + B - A ∩ B
    union_area = calculate_box_area(prediction_box)+calculate_box_area(gt_box)-intersection_area
    iou = intersection_area/union_area
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    # Task 2b
    return 0 if num_tp+num_fp == 0 else num_tp / (num_tp+num_fp)


    raise NotImplementedError


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    return 0 if num_tp+num_fn == 0 else num_tp / (num_tp+num_fn)
    raise NotImplementedError


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # potential_matches contains [pred_box_idx, gt_box_idx, iou]
    potential_matches = np.empty([0,3])
    first = True

    # Find all possible matches with a IoU >= iou threshold
    for pred_box_idx in range(len(prediction_boxes)):
        for gt_box_idx in range(len(gt_boxes)):
            iou = calculate_iou(prediction_boxes[pred_box_idx], gt_boxes[gt_box_idx])
            if (iou >= iou_threshold):
                if (first):
                    potential_matches = np.insert(potential_matches, 0, [pred_box_idx, gt_box_idx, iou], axis=0)
                    first = False
                potential_matches = np.insert(potential_matches, -1, [pred_box_idx, gt_box_idx, iou], axis=0)
                # potential_matches.append([pred_box_idx, gt_box_idx, iou])


    # Sort all matches on IoU in descending order
    sorted_matches = np.sort(potential_matches, axis=0) # defaults to sorting on last axis
    sorted_matches = sorted_matches[::-1] # reverse the list

    # Find all matches with the highest IoU threshold
    
    # Create a list of length number_of_ground_truth_boxes.
    # The list contains the index of the best pred_box match.
    # Iterate through potential matches until all entries in the list are filled.
    # Then, remove the entries in the list containing -1
    max_matches = np.empty(gt_boxes.shape[0], dtype=int)
    max_matches[:] = -1

    num_matched_gt_boxes = 0
    i = 0
    while (num_matched_gt_boxes < gt_boxes.shape[0] and i < sorted_matches.shape[0]):
        gt_box_match_idx = int(sorted_matches[i,1])
        if max_matches[gt_box_match_idx] == -1:
            max_matches[gt_box_match_idx] = sorted_matches[i,0]
            num_matched_gt_boxes += 1
        i += 1
    
    if num_matched_gt_boxes == 0:
        return np.array([]), np.array([])

    gt_box_matches = np.where(max_matches != -1)[0].tolist()
    max_matches = max_matches[~np.all(max_matches == -1)][0].tolist()
    return ([prediction_boxes[i] for i in max_matches], [gt_boxes[i] for i in gt_box_matches])
    # return np.take(prediction_boxes, max_matches, axis=0), np.take(gt_boxes, max_matches, axis=0)



def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    pred_box_matches, gt_box_matches = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)

    true_positives = len(pred_box_matches)
    false_positives = len(prediction_boxes) - true_positives

    false_negatives = len(gt_boxes)-true_positives


    # Find number of matches over the threshold (total number of positives)
    # num_positives = 0
    # for pred_box in prediction_boxes:
    #     for gt_box in gt_boxes:
    #         iou = calculate_iou(pred_box, gt_box)
    #         if (iou >= iou_threshold):
    #             num_positives += 1

    # true_positives = len(pred_box_matches)
    # false_positives = num_positives-true_positives

    # num_predictions = len(prediction_boxes)
    # num_negatives = num_predictions-num_positives
    # true_negatives = num_predictions-true_positives
    # false_negatives = num_negatives-true_negatives

    # true_positives = pred_box_matches.shape[0]
    # false_positives = prediction_boxes.shape[0]-true_positives

    # false_negatives = gt_boxes.shape[0] - true_positives
    # true_negatives = 
    # true_negatives = num_negatives-pred_box_matches.shape[0]
    # false_negatives = num_negatives-true_negatives
    return {"true_pos": true_positives, "false_pos": false_positives, "false_neg": false_negatives}
    raise NotImplementedError


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for img_idx in range(len(all_prediction_boxes)):
        d = calculate_individual_image_result(all_prediction_boxes[img_idx], all_gt_boxes[img_idx], iou_threshold)
        total_tp += d["true_pos"]
        total_fp += d["false_pos"]
        total_fn += d["false_neg"]


    return (calculate_precision(total_tp, total_fp, total_fn), calculate_recall(total_tp, total_fp, total_fn))
    raise NotImplementedError


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)

    # YOUR CODE HERE

    precisions = [] 
    recalls = []


    num_images = all_prediction_boxes
    for thresh_idx in range(len(confidence_thresholds)):
        # Prediction boxes for all images for the current threshold.
        # threshold_predictions[i] holds all the prediction boxes for image i
        threshold_predictions = []

        threshold = confidence_thresholds[thresh_idx]

        for img_idx in range(len(confidence_scores)):
            # Prediction boxes for the current image
            img_predictions = []
            img_scores = confidence_scores[img_idx]
            for score_idx in range(len(img_scores)):
                score = img_scores[score_idx]
                
                if score >= threshold:
                    img_predictions.append(all_prediction_boxes[img_idx][score_idx])
            # scores end
            threshold_predictions.append(img_predictions)
        # img_scores end

        precision, recall = calculate_precision_recall_all_images(threshold_predictions, all_gt_boxes, iou_threshold)
        precisions.append(precision)
        recalls.append(recall)
    # thresholds end
    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    precisions_max_sum = 0

    for lvl in range(len(recall_levels)):
        precision_max = 0

        for n in range(recalls.shape[0]):
    	    if (precisions[n] > precision_max) and (recalls[n] >= recall_levels[lvl]):
                precision_max = precisions[n]

        precisions_max_sum += precision_max

    average_precision = precisions_max_sum / float(len(recall_levels))

    return average_precision




    average_precision = 0
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
