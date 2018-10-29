import numpy as np
from scipy.optimize import minimize

from math import sqrt
import pickle


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    # Numpy handles this better than I do
    return  1/(np.exp(-z) +1)


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - remove features that have the same value for all data points
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    # Preparing the data set
    with open('AI_quick_draw.pickle', 'rb') as open_ai_quick:
        train_data = pickle.load(open_ai_quick)
        train_label = pickle.load(open_ai_quick)
        test_data = pickle.load(open_ai_quick)
        test_label = pickle.load(open_ai_quick)

    
    #Pick a reasonable size for validation data
    
    #Your code here
    # Remove all similar features accross all training data and test data
    all_data = np.concatenate((train_data, test_data))
    same_feature = []
    chosen_feature = []
    for feature in range(784):
        # The feature at every data must be the same as the first data
        feature_val = all_data[0][feature]
        removable = True
        for data in all_data:
            if data[feature] != feature_val:
                # This feature is not the same across all data
                removable = False
                break
        if removable:
            same_feature.append(feature)
        else:
            chosen_feature.append(feature)
    train_data = np.delete(train_data, same_feature, 1)  # np.delete does not occur in place
    test_data = np.delete(test_data, same_feature, 1)
    print("number of unused feature", len(same_feature))
    print("size of training data after feature selection", train_data.shape)
    # Normalizing all data to [0-1]
    train_data.astype(float)
    train_data = train_data / np.amax(train_data)
    test_data.astype(float)
    test_data = test_data / np.amax(test_data)
    print("data normalized to [0-1]")

    # Shuffle data up
    permutation = np.random.permutation(train_data.shape[0])
    train_data = train_data[permutation]
    train_label = train_label[permutation]
    print("All data shuffled randomly")
    # Set validation set
    validation_size = 10000
    validation_data = train_data[0:validation_size, :]
    validation_label = train_label[0:validation_size]
    train_data = np.delete(train_data, [index for index in range(0, validation_size)], 0)
    train_label = np.delete(train_label, [index for index in range(0, validation_size)], 0)
    print("Validation set", validation_data.shape)
    print("Trainning set", train_data.shape)


    print ("preprocess done!")

    return train_data, train_label, validation_data, validation_label, test_data, test_label,chosen_feature


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Your code here
    # Need to calculate delta_l first, as it is used for both the output and hidden gradiant
    # The gradient is calculated as the sum of error over the entire dataset.
    # Batch gradient descent

    # Calculate the obj_val
    # Doing the feed forward
    data = np.append(training_data, np.ones(shape=(training_data.shape[0], 1)), axis=1)
    hidden = sigmoid(np.matmul(data, w1.transpose()))
    hidden = np.append(hidden, np.ones(shape=(hidden.shape[0], 1)), axis=1)
    output = sigmoid(np.matmul(hidden, w2.transpose()))

    # Done with the feed forward
    # Need to make it into one-hot version
    label = np.zeros(shape=(training_label.shape[0], n_class))
    for i in range(training_label.shape[0]):
        label[i][int(training_label[i])] = 1
    # Made the one-hot version
    # The object function
    obj_val = -1 / (1.0 * training_data.shape[0]) * np.sum(np.multiply(label, np.log(output)) +
                                                           np.multiply((1 - label), np.log(1 - output))) + \
              lambdaval / (2.0 * training_data.shape[0]) * (np.sum(np.power(w1, 2)) + np.sum(np.power(w2, 2)))

    # # Calculate delta_l
    delta_l = np.subtract(output, label)
    # grad_2
    grad_w2 = 1 / (training_data.shape[0]) * (np.matmul(delta_l.transpose(), hidden) + lambdaval * w2)

    scalar_sum = np.matmul(delta_l, w2[:, 0:n_hidden])
    z_sum = np.multiply(1 - hidden[:, 0:n_hidden], hidden[:, 0:n_hidden])
    z_sum = np.multiply(z_sum, scalar_sum)
    grad_w1 = (1 / training_data.shape[0]) * (np.matmul(z_sum.transpose(), data) + lambdaval * w1)
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)
    print(obj_val)
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""
    # Data is suppose to be (number_of_data,picture_size), so we need to add a bias term
    data = np.append(data, np.ones(shape=(data.shape[0], 1)), axis=1)
    hidden = sigmoid(np.matmul(data, w1.transpose()))
    hidden = np.append(hidden, np.ones(shape=(hidden.shape[0], 1)), axis=1)
    output = sigmoid(np.matmul(hidden, w2.transpose()))
    # Now the output is (_number_of_data,10 ) with 10 as in number of class
    # np.argmax return the labels from 0-9, need to add 1 in
    labels = np.argmax(output, axis=1)

    # Your code here
    return labels.reshape((labels.shape[0], 1))


"""**************Neural Network Script Starts here********************************"""
if __name__=="__main__":
    train_data, train_label, validation_data, validation_label, test_data, test_label,chosen_feature = preprocess()

    #  Train Neural Network

    # set the number of nodes in input unit (not including bias unit)
    n_input = train_data.shape[1]

    # set the number of nodes in hidden unit (not including bias unit)
    n_hidden = 150

    # set the number of nodes in output unit
    n_class = 10

    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    # set the regularization hyper-parameter
    lambdaval = 0

    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {'maxiter': 100}  # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
    # and nnObjGradient. Check documentation for this function before you proceed.
    # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Test the computed parameters

    predicted_label = nnPredict(w1, w2, train_data)

    # find the accuracy on Training Dataset
    train_label.astype(int)
    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, validation_data)

    # find the accuracy on Validation Dataset
    validation_label.astype(int)
    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, test_data)

    # find the accuracy on Validation Dataset
    test_label.astype(int)
    print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
