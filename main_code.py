"""                   
https://github.com/Mohammad-Rahmdel/deep-learning-specialization-coursera/blob/master/02-Improving-Deep-Neural-Networks/week3/Programming-Assignments/Tensorflow_Tutorial.ipynb
"""

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignoring warnings





# def sigmoid(z):
#     x = tf.placeholder(tf.float32, name = 'x')
#     results = tf.sigmoid(x)
#     with tf.Session() as session:
#         results = session.run(results, feed_dict={x: z})    
#     return results


# def cost(logits, labels):
#     """
#     Computes the cost using the sigmoid cross entropy
    
#     Arguments:
#     logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
#     labels -- vector of labels y (1 or 0) 
    
#     Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels" 
#     in the TensorFlow documentation. So logits will feed into z, and labels into y. 
    
#     Returns:
#     cost -- runs the session of the cost 
#     """
#     z = tf.placeholder(tf.float32, name='z')
#     y = tf.placeholder(tf.float32, name='y')

#     J = tf.nn.sigmoid_cross_entropy_with_logits(logits = z,  labels = y)

#     with tf.Session() as session:
#         cost = session.run(J, feed_dict={z: logits, y: labels})

    
#     return cost


# def one_hot_matrix(labels, C):
#     """
#     Creates a matrix where the i-th row corresponds to the ith class number and the jth column
#                      corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
#                      will be 1. 
                     
#     Arguments:
#     labels -- vector containing the labels 
#     C -- number of classes, the depth of the one hot dimension
    
#     Returns: 
#     one_hot -- one hot matrix
#     """

#     one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)
#     with tf.Session() as session:
#         one_hot_matrix = session.run(one_hot_matrix)

#     return one_hot_matrix



# def ones(shape):
#     """
#     Creates an array of ones of dimension shape
    
#     Arguments:
#     shape -- shape of the array you want to create
        
#     Returns: 
#     ones -- array containing only ones
#     """

#     ones = tf.ones(shape)
#     with tf.Session() as session:
#         ones = session.run(ones)

#     return ones


# Problem statement: SIGNS Dataset 

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# # show image[index]
# index = 4
# plt.imshow(X_train_orig[index])
# print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
# plt.show()


# Flattening
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalizing
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Converting labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)



def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """
    X = tf.placeholder(shape=[n_x, None] ,dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=[n_y, None] ,dtype=tf.float32, name='Y')
    return X, Y




def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
    W1 : [25, 12288], b1 : [25, 1], W2 : [12, 25], b2 : [12, 1], W3 : [6, 12], b3 : [6, 1]

    Returns:
    parameters 
    """

    tf.set_random_seed(1)                  

    b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
    b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
    b3 = tf.get_variable("b3", [6,1], initializer = tf.zeros_initializer())
    # using Xavier Initialization for weights
    W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    W3 = tf.get_variable("W3", [6,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


def forward_propagation(X, parameters, keep_prob=1):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X) , b1)
    A1 = tf.nn.relu(Z1)
    A1 = tf.nn.dropout(A1, keep_prob) # adding dropout did not work!
    Z2 = tf.add(tf.matmul(W2, A1) , b2)
    A2 = tf.nn.relu(Z2)
    A2 = tf.nn.dropout(A2, keep_prob)
    Z3 = tf.add(tf.matmul(W3, A2) , b3)

    return Z3


    
def compute_cost(Z3, Y):
    """
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    logits = tf.transpose(Z3)   # shape (number of examples, num_classes)
    labels = tf.transpose(Y)    # shape (number of examples, num_classes)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    # tf.reduce_mean basically does the summation over the examples

    return cost



def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train (input size = 12288, number of training examples = 1080)
    Y_train (output size = 6, number of training examples = 1080)
    X_test (12288, 120)
    Y_test (6, 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                    
    
    m = Y_train.shape[1]
    n_x = X_train.shape[0]
    n_y = Y_train.shape[0]
    X, Y = create_placeholders(n_x, n_y)
    costs = [] 

    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer() # Initialize all the variables

    with tf.Session() as session:
        session.run(init)
        for i in range(num_epochs):
            # seed = seed + 1
            epoch_cost = 0.
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            n_minibatches = np.floor(m / minibatch_size)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = session.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / n_minibatches

            
            if print_cost == True:
                if i % 100 == 0:
                    print ("Cost after epoch %i: %f" % (i, epoch_cost))
                if i % 5 == 0:
                    costs.append(epoch_cost)
            
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()



        parameters = session.run(parameters)
        print ("Parameters have been trained!")
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

    return parameters
    

parameters = model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32)

""" Results before Regularization 

learning_rate = 0.0001
num_epochs = 1500
minibatch_size = 32

with seed = seed + 1 :
    Train Accuracy: 0.9990741
    Test Accuracy: 0.71666664

without seed = seed + 1 :
    Train Accuracy: 0.9962963
    Test Accuracy: 0.8333333


learning_rate = 0.0001
num_epochs = 1000
minibatch_size = 32
    Train Accuracy: 0.9824074
    Test Accuracy: 0.81666666

learning_rate = 0.00005
num_epochs = 1500
minibatch_size = 32
    Train Accuracy: 0.99814814
    Test Accuracy: 0.81666666


learning_rate = 0.0001
num_epochs = 1500
minibatch_size = 64
    Train Accuracy: 0.9925926
    Test Accuracy: 0.81666666


learning_rate = 0.0001
num_epochs = 2500
minibatch_size = 32
    Train Accuracy: 0.9814815
    Test Accuracy: 0.8333333

learning_rate = 0.0001
num_epochs = 1500
minibatch_size = 16
    Train Accuracy: 0.97314817
    Test Accuracy: 0.725
    time = 7mins

learning_rate = 0.001, 0.0003
OVERSHOOTING occures!!!


"""









# # Exercise 2.7 - Test with your own image
# import scipy
# from PIL import Image
# from scipy import ndimage

# ## START CODE HERE ## (PUT YOUR IMAGE NAME) 
# my_image = "victory.jpeg"
# ## END CODE HERE ##

# # We preprocess your image to fit your algorithm.
# fname = "images/" + my_image
# image = np.array(ndimage.imread(fname, flatten=False))
# my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
# my_image_prediction = predict(my_image, parameters)

# plt.imshow(image)
# print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
# # Your algorithm predicts: y = 1!  incorrect answer


"""
    Tensorflow is a programming framework used in deep learning
    The two main object classes in tensorflow are Tensors and Operators.
    When you code in tensorflow you have to take the following steps:
        Create a graph containing Tensors (Variables, Placeholders ...) and Operations (tf.matmul, tf.add, ...)
        Create a session
        Initialize the session
        Run the session to execute the graph
    You can execute the graph multiple times as you've seen in model()
    The backpropagation and optimization is automatically done when running the session on the "optimizer" object.
"""



