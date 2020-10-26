"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    for i in range(np.shape(X)[0]):
        probs = np.exp(X[i].dot(W)) / np.sum(np.exp(X[i].dot(W)))
        loss += -np.log(probs[y[i]])

        deriv = probs
        deriv[y[i]] -= 1
        dW += np.dot(X[i].reshape(-1, 1), deriv.reshape(1, -1))

    # for i in range(len(y)):
    #     for j in range(np.shape(W)[1]):
    #         for k in range(np.shape(W)[0]):
    #             if(j==y[i]):
    #                 dW[k][j] += (np.exp(np.dot(X[i], W.T[j])) - 1) * X[i][k] / den[i]
    #             else:
    #                 dW[k][j] += np.exp(np.dot(X[i], W.T[j])) * X[i][k] / den[i]
    loss /= np.shape(X)[0]
    loss += 0.5 * reg * np.sum(W * W)
    dW /= np.shape(X)[0]
    dW += reg * W

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    scores = np.dot(X, W)
    exp_scores = np.exp(scores)

    norm = np.sum(exp_scores, axis=1, keepdims=True)
    prob = exp_scores / norm
    loss = 0.5 * reg * np.sum(W * W) + np.sum(-np.log(prob[range(np.shape(X)[0]), y])) / np.shape(X)[0]

    deriv = prob
    deriv[range(np.shape(X)[0]), y] -= 1
    dW = np.dot(X.T, deriv) / np.shape(X)[0] + reg * W
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 1e-6]
    regularization_strengths = [5e3, 5e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################

    for lr in learning_rates:
        for rs in regularization_strengths:
            LC = LinearClassifier()
            loss_history = LC.train(X_train, y_train, learning_rate=lr, reg=rs, num_iters=1000, verbose=True)
            y_train_pred = LC.predict(X_train)
            y_val_pred = LC.predict(X_val)
            all_classifiers.append([LC,np.mean(y_val == y_val_pred)])
            if np.mean(y_val == y_val_pred) > best_val:
                best_val = np.mean(y_val == y_val_pred)
                best_softmax = LC

            results[(lr, rs)] = (np.mean(y_train == y_train_pred), np.mean(y_val == y_val_pred))
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
