from math import floor, sqrt

from scipy.special import comb


def threshold_classifier(trigger_size, number_labels, error_rate=0.001):
    """Compute threshold for classification task.

    Args:
        trigger_size (int): Number of trigger instances
        number_labels (int): Number of labels
        error_rate (int): Error rate of verification

    Returns:
        threshold (float): The minimal threshold

    """
    # Define parameters
    threshold = 1 / number_labels
    precision = 1 / trigger_size
    S = 0

    # Compute confidence level

    for i in range(0, int(threshold * trigger_size) + 1):
        wrong_detected = ((1 - 1 / number_labels)**(trigger_size - i))
        S += comb(trigger_size, i) * (1 / (number_labels**i)) * wrong_detected
    # Continue while confidence < 1 -e
    while S < (1 - error_rate) or threshold == 1:
        old_threshold = threshold
        threshold += precision
        old_bound = floor(old_threshold * trigger_size) + 1
        new_bound = floor(threshold * trigger_size) + 1
        for i in range(old_bound, new_bound):
            wrong_detected = (1 - 1 / number_labels)**(trigger_size - i)
            S += comb(trigger_size, i) * \
                (1 / number_labels**i) * wrong_detected

    return min(threshold, 1)


def threshold_RMSE(upper_bound, lower_bound, q=3):
    """Compute threshold for RMSE metric.

    Args:
        upper_bound (int): Upper bound of regression outputs
        lower_bound (int): lower bound of regression outputs
        q (int): Quantification parameter

    Returns:
        threshold (float): The minimal threshold

    """

    return (upper_bound - lower_bound) / q


def threshold_MAPE(upper_bound, lower_bound, q=3):
    """Compute threshold for MAPE metric.

    Args:
        upper_bound (int): Upper bound of regression outputs
        lower_bound (int): lower bound of regression outputs
        q (int): Quantification parameter

    Returns:
        threshold (float): The minimal threshold

    """
    return (upper_bound - lower_bound) / q * upper_bound


def verify(outputs_original,
           outputs_suspect,
           bounds=None,
           number_labels=None,
           error_rate=0.001,
           metric='accuracy'):
    """Verify watermark based on trigger outputs.

    Args:
        outputs_original (array): Predictions original
        outputs_suspect (array): Predictions suspect
        bounds (tuples): Bounds for threshold (regression)
        number_labels (int): Number of labels (classification)
        error_rate (int): Error rate of verification

    Returns:
        is_stolen (bool): Is the model stolen ?

    """

    # Compute threshold
    is_stolen = None
    trigger_size = len(outputs_suspect)

    # Verification for each of the metrics
    if metric == 'accuracy':
        # Compute threshold
        threshold = threshold_classifier(trigger_size,
                                         number_labels,
                                         error_rate=1e-6)
        accuracy = 0
        for i, j in zip(outputs_original, outputs_suspect):
            accuracy += int(i == j)
        accuracy = accuracy / trigger_size
        is_stolen = accuracy >= threshold

    elif metric == 'RMSE':
        lower_bound, upper_bound = bounds
        # Compute threshold
        threshold = threshold_RMSE(upper_bound, lower_bound)
        rmse_score = 0
        for i, j in zip(outputs_original, outputs_suspect):
            rmse_score += (i - j)**2
        rmse_score = sqrt(rmse_score / len(outputs_suspect))
        is_stolen = rmse_score <= threshold

    elif metric == 'MAPE':
        lower_bound, upper_bound = bounds
        # Compute threshold
        threshold = threshold_MAPE(upper_bound, lower_bound)
        mape_score = 0
        for i, j in zip(outputs_original, outputs_suspect):
            mape_score += abs(i - j) / i
        mape_score = mape_score / len(outputs_suspect)
        is_stolen = mape_score <= threshold

    return is_stolen
