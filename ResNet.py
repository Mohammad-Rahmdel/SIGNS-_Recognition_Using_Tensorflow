"""                   
In recent years, neural networks have become deeper, with state-of-the-art networks going from just a few layers 
(e.g., AlexNet) to over a hundred layers.

The main benefit of a very deep network is that it can represent very complex functions. 
It can also learn features at many different levels of abstraction, from edges (at the lower layers) 
to very complex features (at the deeper layers). However, using a deeper network doesn't always help.
"""

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


from keras import regularizers



def identity_block(X, f, filters, stage, block, lambd=0.0):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', 
    kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(lambd))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path 
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', 
    kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(lambd))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', 
    kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(lambd))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)

    return X



def convolutional_block(X, f, filters, stage, block, s = 2, lambd=0.0):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0), 
    kernel_regularizer=regularizers.l2(lambd))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path 
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', 
    kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(lambd))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', 
    kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(lambd))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)


    ##### SHORTCUT PATH #### 
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', 
    kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X




def ResNet50(input_shape = (64, 64, 3), classes = 6, lambd=0.0, keep_prob=1.0):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, kernel_size = (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0), 
    kernel_regularizer=regularizers.l2(lambd))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1, lambd=lambd)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b', lambd=lambd)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c', lambd=lambd)

    # Stage 3 
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2, lambd=lambd)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b', lambd=lambd)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c', lambd=lambd)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d', lambd=lambd)

    # Stage 4 
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2, lambd=lambd)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b', lambd=lambd)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c', lambd=lambd)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d', lambd=lambd)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e', lambd=lambd)

    # Stage 5 
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2, lambd=lambd)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c', lambd=lambd)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='d', lambd=lambd)

    # AVGPOOL 
    X = AveragePooling2D(pool_size=(2,2), name='avg_pool')(X)
    
    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    X = Dropout(keep_prob)(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model


model = ResNet50(input_shape = (64, 64, 3), classes = 6, lambd=0.0, keep_prob=1.0)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T



model.fit(X_train, Y_train, epochs = 2, batch_size = 32)

preds = model.evaluate(X_train, Y_train)
print ("Test Accuracy = " + str(preds[1]))

preds = model.evaluate(X_test, Y_test)
print ("Test Accuracy = " + str(preds[1]))



















""" RESLUTS :
USING CPU
epochs = 2, batch_size = 32
Train Accuracy = 0.7388888888888889
Test Accuracy = 0.724999996026357

USING GPU
epochs = 20, batch_size = 32
#1
Test Accuracy = 0.9777777777777777
Test Accuracy = 0.9083333373069763
#2
Test Accuracy = 0.987037037037037
Test Accuracy = 0.949999996026357

epochs = 12, batch_size = 32
Test Accuracy = 0.9814814814814815
Test Accuracy = 0.900000003973643

epochs = 30, batch_size = 32
Test Accuracy = 0.9740740740740741
Test Accuracy = 0.925000003973643

epochs = 30, batch_size = 32
lambda=1e-5
Test Accuracy = 0.9879629629629629
Test Accuracy = 0.9416666626930237

epochs = 20, batch_size = 32
keep_prob = 0.9 lambda=0
drop out after FC
awful

epochs = 20, batch_size = 32
keep_prob = 0.9 lambda=0
drop out before FC, after dense
Test Accuracy = 0.9796296300711456
Test Accuracy = 0.9416666666666667

epochs = 50, batch_size = 32
keep_prob = 0.9 lambda=1e-5
drop out before FC, after dense
Test Accuracy = 0.9935185185185185
Test Accuracy = 0.95




Very deep "plain" networks don't work in practice because they are hard to train due to vanishing gradients.
The skip-connections help to address the Vanishing Gradient problem. They also make it easy for a ResNet block to learn 
an identity function.
There are two main type of blocks: The identity block and the convolutional block.
Very deep Residual Networks are built by stacking these blocks together.

"""