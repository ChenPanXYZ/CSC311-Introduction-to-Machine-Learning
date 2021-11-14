from utils import sigmoid

import numpy as np


def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    #####################################################################
    # TODO:                                                             #
    # Given the weights and bias, compute the probabilities predicted   #
    # by the logistic classifier.                                       #
    #####################################################################
    
    y = None
    N = len(data)
    M = len(weights) - 1
    temp = np.ones([N, M + 1])
    temp[: N, : M] = np.array(data)
    data = temp



    # print(np.matmul(data, weights).shape)
    z = np.matmul(data, weights)
    y = sigmoid(z)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return y


def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    #####################################################################
    # TODO:                                                             #
    # Given targets and probabilities predicted by the classifier,      #
    # return cross entropy and the fraction of inputs classified        #
    # correctly.                                                        #
    #####################################################################

    ce = None
    frac_correct = None
    assert len(targets) == len(y)
    N = len(targets)

    ce = - np.mean([
        targets.flatten()[i] * np.log(y.flatten()[i]) + (1 - targets.flatten()[i]) * np.log(1 - y.flatten()[i]) for i in range(0, N)
    ])

    # TODO: why -targets is something like 255?

    # TODO: Threshold

    frac_correct = np.mean([1 if targets[i][0] == np.round(y[i][0]) else 0 for i in range(0, N)])

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)

    #####################################################################
    # TODO:                                                             #
    # Given weights and data, return the averaged loss over all data    #
    # points, gradient of parameters, and the probabilities given by    #
    # logistic regression.                                              #
    #####################################################################
    f = None
    df = None

    f = evaluate(targets, y)[0]

    N = len(data)
    M = len(weights) - 1   
    temp = np.ones([N, M + 1])
    temp[: N, : M] = np.array(data)


    df = np.zeros([M+1, 1])

    df[:, 0] = np.array([[np.mean([(y.flatten()[i] - targets.flatten()[i]) * temp[i][j] for i in range(0, N)]) for j in range(0, M + 1)],])

    # df = np.matrix([[np.mean([(y[i] - targets[i]) * temp[i][j] for i in range(0, N)]) for j in range(0, M + 1)],])

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y
