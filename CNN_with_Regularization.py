import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *


np.random.seed(1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# # Example of a picture
# index = 21
# plt.imshow(X_train_orig[index])
# print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
# plt.show()



X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

conv_layers = {}





def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True, lambd=0.0, keep_prob=1):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X_train (None, 64, 64, 3)
    Y_train (None, n_y = 6)
    X_test  (None, 64, 64, 3)
    Y_test  (None, n_y = 6)

    """
    
    ops.reset_default_graph()      # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             
    seed = 3                                          
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = []  
    train_acc = []
    test_acc = []                                   


    X = tf.placeholder(shape=[None, n_H0, n_W0, n_C0], dtype=tf.float32, name = 'X')
    Y = tf.placeholder(shape=[None, n_y], dtype=tf.float32, name = 'Y')

    """
    X = [none, 64, 64, 3]
    W1 = [4, 4, 3, 8]
    W2 = [2, 2, 8, 16]
    """
    tf.set_random_seed(1)                        
    W1 = tf.get_variable("W1", [4,4,3,8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [2,2,8,16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))


    s = 1
    Z1 = tf.nn.conv2d(X, W1, strides = [1,s,s,1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    f = s = 8 # f = window size
    P1 = tf.nn.max_pool(A1, ksize = [1,f,f,1], strides = [1,s,s,1], padding = 'SAME')


    s = 1
    Z2 = tf.nn.conv2d(P1, W2, strides = [1,s,s,1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    f = s = 4
    P2 = tf.nn.max_pool(A2, ksize = [1,f,f,1], strides = [1,s,s,1], padding = 'SAME')

    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)

    # FULLY-CONNECTED without non-linear activation function (not call softmax).
    # 6 neurons in output layer. "activation_fn=None" 
    Z3 = tf.contrib.layers.fully_connected(P2, num_outputs=6, activation_fn=None)
    Z3 = tf.nn.dropout(Z, keep_prob)

    """
    In the last function above (tf.contrib.layers.fully_connected), the fully connected layer automatically 
    initializes weights in the graph and keeps on training them as you train the model. Hence, you did not need 
    to initialize those weights when initializing the parameters.
    """

    cost = tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y) + lambd * tf.nn.l2_loss(W1) + lambd * tf.nn.l2_loss(W2)
    
    cost = tf.reduce_mean(cost)
    
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer() 
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
            
                minibatch_cost += temp_cost / num_minibatches



            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)


            if epoch % 1 == 0 :
                predict_op = tf.argmax(Z3, 1)
                correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
                
                # Calculate accuracy on the test set
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                train_acc.append(accuracy.eval({X: X_train, Y: Y_train}))
                test_acc.append(accuracy.eval({X: X_test, Y: Y_test}))
            
        _, ax = plt.subplots()
        ax.plot(np.squeeze(train_acc), '-b', label='train accuracy')
        ax.plot(np.squeeze(test_acc), '-r', label='test accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
             

        
        
        
        # # plot the cost
        # plt.plot(np.squeeze(costs))
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per tens)')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

    
model(X_train, Y_train, X_test, Y_test, learning_rate = 0.005, num_epochs = 5, minibatch_size = 32, print_cost = True)




""" 
Results before adding Regularization :

learning_rate = 0.003
num_epochs = 150
minibatch_size = 64
Train Accuracy: 0.97037035
Test Accuracy: 0.85833335


learning_rate = 0.003
num_epochs = 100
minibatch_size = 64
Train Accuracy: 0.91944444
Test Accuracy: 0.8666667


learning_rate = 0.005
num_epochs = 150
minibatch_size = 64
Train Accuracy: 0.9898148
Test Accuracy: 0.875

learning_rate = 0.005
num_epochs = 150
minibatch_size = 32
Train Accuracy: 0.98055553
Test Accuracy: 0.85833335


learning_rate = 0.005
num_epochs = 200
minibatch_size = 32
Train Accuracy: 1.0
Test Accuracy: 0.925


learning_rate = 0.005
num_epochs = 250
minibatch_size = 32
Train Accuracy: 1.0
Test Accuracy: 0.9166667


learning_rate = 0.005
num_epochs = 500
minibatch_size = 32
Train Accuracy: 1.0
Test Accuracy: 0.85




Results after adding L2 Regularization :


learning_rate = 0.005
num_epochs = 200
minibatch_size = 32
lambda = 0.01
Train Accuracy: 0.9527778
Test Accuracy: 0.85

learning_rate = 0.005
num_epochs = 200
minibatch_size = 32
lambda = 0.0001
Train Accuracy: 0.96666664
Test Accuracy: 0.7916667


learning_rate = 0.003
num_epochs = 200
minibatch_size = 32
lambda = 0.001
Train Accuracy: 0.98796296
Test Accuracy: 0.9166667


Results after adding Dropout to FC layer:

learning_rate = 0.003
num_epochs = 100
minibatch_size = 32
keep_prob = 0.5
Train Accuracy: 0.6074074
Test Accuracy: 0.6333333


learning_rate = 0.004
num_epochs = 200
minibatch_size = 32
keep_prob = 0.9
Train Accuracy: 0.94166666
Test Accuracy: 0.875


learning_rate = 0.003
num_epochs = 120
minibatch_size = 32
keep_prob = 0.9
Train Accuracy: 0.9351852
Test Accuracy: 0.89166665


learning_rate = 0.005
num_epochs = 220
minibatch_size = 32
keep_prob = 0.75
Train Accuracy: 0.8037037
Test Accuracy: 0.8666667


learning_rate = 0.003
num_epochs = 500
minibatch_size = 32
keep_prob = 0.9
Train Accuracy: 0.9546296
Test Accuracy: 0.94166666

learning_rate = 0.003
num_epochs = 500
minibatch_size = 32
keep_prob = 0.9
lambda = 0.001
Train Accuracy: 0.962037
Test Accuracy: 0.93333334


@@@@@@@@@@@@@@@@@@@@@@@ look at this @@@@@@@@@@@@@@@@@@@@@@
learning_rate = 0.002
num_epochs = 500
minibatch_size = 32
keep_prob = 0.9
lambda = 0
Train Accuracy: 0.962037
Test Accuracy: 0.94166666

@@@@@@@@@@@@@@@@@@ compare with this one @@@@@@@@@@@@@@@@@@@@@
learning_rate = 0.002
num_epochs = 500
minibatch_size = 32
keep_prob = 1
lambda = 0
Train Accuracy: 1.0
Test Accuracy: 0.8833333

Drop out influence :)
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

Applying dropout to the final fully-connected layers effectively ensemble the entire network, 
including all previous layers.

Dropout has the added benefit of reducing dependencies within each layer so it can be beneficial to apply to all of them. 
For example, in the original paper it was applied to convolutional layers in addition to fully-connected layers and showed 
improvement over just applying dropout to the fully-connected layers. However, dropout can increase training time so it is 
usually omitted from the convolutional layers.

Furthermore, batch normalization largely removes the need for dropout (see bn paper) but, due to the size of the dataset 
here, keeping dropout on the fully-connected layers (which don't use bn) was helpful.
"""