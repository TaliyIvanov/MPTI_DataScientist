import numpy as np
def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    # TODO: implement metrics
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    true_positives = np.sum(np.logical_and(prediction, ground_truth))
    false_positives = np.sum(np.logical_and(prediction, np.logical_not(ground_truth)))
    true_negatives = np.sum(np.logical_and(np.logical_not(prediction), np.logical_not(ground_truth)))
    false_negatives = np.sum(np.logical_and(np.logical_not(prediction), ground_truth))
    
    # Calculate precision
    precision = true_positives / (true_positives + false_positives)
    
    # Calculate recall
    recall = true_positives / (true_positives + false_negatives)
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # Calculate accuracy
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives +
                                                    false_negatives)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    if prediction.shape != ground_truth.shape:
        raise ValueError("Shape mismatch between prediction and ground_truth arrays")
    correct_predictions = np.sum(prediction == ground_truth)
    total_samples = prediction.shape[0]
    accuracy = correct_predictions / total_samples

    return accuracy
