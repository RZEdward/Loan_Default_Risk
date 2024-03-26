# Logistic Regression from Scratch

# Linear Regression: y_pred = wx + b
# Logistic Regression: y_prob = 1 / ( 1 + exp (-wx+b)) ) = h_theta_(x)
# Logistic Regression uses cross-entropy as an alternative to error: J(w,b) = J(theta) = ... -
# Cross-Entropy derived from MLE principles
# Then take dJ/dw and dJ/db and place them in a (2,1) vector
# We want to minimise J(theta) so use gradient descent to optimise w, b and find J_min
# w =; w - alpha*dJ/dw
# b =; b - alpha*dJ/db
# alpha = learning rate, set to 0.01 to begin with
# need to define number of iterations == n_iters
# An epoch elapses when an entire dataset is passed forward and backward through the neural network exactly one time.  If the entire dataset cannot be passed into the algorithm at once, it must be divided into mini-batches.  Batch size is the total number of training samples present in a single min-batch.  An iteration is a single gradient update (update of the model's weights) during training.  The number of iterations is equivalent to the number of batches needed to complete one epoch.  
# So if a dataset includes 1,000 images split into mini-batches of 100 images, it will take 10 iterations to complete a single epoch.
# but if you just have 1 batch (small dataset) then 1 epoch = 1 iteration


# Optimise s(z) such that samples are optimally separated around s(z) = 0.5
# s = sigmoid function
# z = Theta.T * X

# X = ( [1, X1, X2, X3]^n ) - each row is the set of features corresponding to 1 sample
# Theta = ( Theta0, Theta1, Theta2, Theta3  )
# Y = ( {0,1}^n ) - classification labels

# What is our cost function / fit modifier?

import numpy as np
import math
from sklearn.metrics import roc_auc_score  # only for AUC calc


def sigmoid(x):
    return 1/(1 + np.exp(-x) )

class LogisticReg():
    
    def __init__(self, alpha=0.001, n_iters=1000):

        self.learning_rate = alpha
        self.n_iters = n_iters
        self.weights = None        # do we need to declare w, b here?
        self.bias = None

    def split(self, X, Y, test_size, random_state=1234, Joined=True):   # if joined == true, then we just have one matrix with labels appended to samples, else we have X and Y separately

        if Joined == True:
            n_samples, n_features = X.shape
            Y = [X[i][-1] for i in range(len(X[n_features]))]
            X = [X[i][0:-1] for i in range(len(X[n_features]))]

        n_samples, n_features = X.shape
        len_test = round(test_size*n_samples)
        len_train = n_samples - len_test
        X_test = X[0:len_test]
        X_train = X[len_test:]
        Y_test = Y[0:len_test]
        Y_train = Y[len_test:]
        # could select len_test samples from random elements in X

    def fit(self, X, Y):    # input X, Y as X_train, Y_train
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) # initialize all weights = 0, in array of shape ( n_features, )
        self.bias = 0                       # initialize bias as 0
        
        # histoire  -  list of lists of all params at each iteration/epoch

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            pred = sigmoid(linear_pred)
            dw = 1/n_samples * np.dot( X.T, ( pred - Y ) )  # Y is in form (n_samples,) , X in form (n_samples, n_features) - so need to take transpose to align dimensions - as you are just taking a broad mean of all dif_y*x_i it doesnt matter conceptually
            db = 1/n_samples * np.sum( pred - Y )
            self.weights -= self.learning_rate*dw
            self.bias -= self.learning_rate*db

    def predict(self, X):   # input X as X_test
        
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_predictions = [ 0 if y < 0.5 else 1 for y in y_pred ]
        return class_predictions
    
    def predict_proba(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        probabilities = sigmoid(linear_pred)
        return probabilities

    def stats(self, X, Y):
        Y = np.array(Y)  # Ensure Y is a Numpy array
        Y_pred = np.array(self.predict(X))
        Y_pred_proba = self.predict_proba(X)

        TP = np.sum((Y == 1) & (Y_pred == 1))
        FP = np.sum((Y == 0) & (Y_pred == 1))
        FN = np.sum((Y == 1) & (Y_pred == 0))

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        accuracy = np.sum(Y_pred == Y) / len(Y)
        
        # Calculate AUC-ROC using sklearn, ensuring Y and Y_pred_proba are compatible
        AUC = roc_auc_score(Y, Y_pred_proba)

        return [accuracy, precision, recall, F1, AUC]


    def visual_analytics(self, X_Train, Y_Train, X_Test, Y_Test):      
        
        # at each iteration of fit(X_Train, Y_Train), record w, b, cross-entropy, train_acc, test_acc
        # just modify fit() to return list of lists of each of these
        # then in visual_analytics() plot iteration vs index of each list
        # call stats(X_Test, Y_Test) for final stats
        # create dashboard of plots + table of stats + confusion matrix
    
        pass