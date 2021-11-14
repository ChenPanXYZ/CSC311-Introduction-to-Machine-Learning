from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.3,
        "weight_regularization": 0.,
        "num_iterations": 200
    }
    weights = np.zeros((M + 1, 1))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    train_ce = []
    valid_ce = []
    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
        weights = weights - hyperparameters["learning_rate"] * df

        evaluate_result = evaluate(train_targets, y)
        if t == hyperparameters["num_iterations"] - 1:
            print(evaluate_result[0])
        train_ce.append(evaluate_result[0])

        f, df, y = logistic(weights, valid_inputs, valid_targets, hyperparameters)
        evaluate_result = evaluate(valid_targets, y)
        valid_ce.append(evaluate_result[0])

    # validation set
    f, df, y = logistic(weights, valid_inputs, valid_targets, hyperparameters)
    print(evaluate(valid_targets, y)[0])

    # Selected best models, can see the errors.
    f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
    print("Train Set:")
    print(evaluate(train_targets, y))

    f, df, y = logistic(weights, valid_inputs, valid_targets, hyperparameters)
    print("Validation Set:")
    print(evaluate(valid_targets, y))


    # Plot
    iterations = [t for t in range(len(train_ce))]
    plt.plot(iterations, train_ce, label = "Train")
    plt.plot(iterations, valid_ce, label = "Validation")
    plt.xlabel("Iteration")
    plt.ylabel("Cross-entropy")
    plt.title("Cross-entropy as train progresses (Large Train)")
    plt.legend()
    # plt.xticks(np.arange(hyperparameters["num_iterations"]), np.arange(1, hyperparameters["num_iterations"]+1))
    plt.savefig("q32b-large.png")




    # After find the best model

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
    # run_pen_logistic_regression()
