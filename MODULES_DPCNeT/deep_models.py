
# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


def DPNNet_build(dim, regress=False):
    # define our MLP network
    model = Sequential()
    model.add(Dense(256, input_dim=dim, activation="relu", kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dense(4, activation="relu", kernel_regularizer=regularizers.l2(0.0001)))

    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))

#     optimizer = tf.keras.optimizers.RMSprop(0.0001) # the optimizer used is RMSprop, one can use SGD(stochastic Gradient decent)
# #   optimizer = tf.keras.optimizers.Adam(0.001) #Adam(lr=1e-3, decay=1e-3 / 200)

#     model.compile(loss='mean_squared_error',
#                 optimizer=optimizer,
#                 metrics=['mean_absolute_error', 'mean_squared_error'])
#     model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

    # return our model
    return model


def build_cnn(width, height, depth, filters=(16, 32, 64, 128), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1

    # define the model input
    inputs = Input(shape=inputShape)

    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs

        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)

    # check to see if the regression node should be added
    if regress:
        #         x = Dense(1, activation="linear")(x)
        x = Dense(1, activation="linear")(x)

    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model
