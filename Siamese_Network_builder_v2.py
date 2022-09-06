import math

import tensorflow as tf
from isapi.install import verbose
from keras import Model
from keras.layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Input, Lambda, BatchNormalization, Dropout
from keras.activations import sigmoid
from keras import optimizers

class Siamese_Network_builder_v2:
    def __init__(self, num_of_filters, save_path):
        self.num_of_filters = num_of_filters
        self.save_path = save_path
        pass

    def difference_layer(self, tensor_set):
        tensor1 = tensor_set[0]
        tensor2 = tensor_set[1]
        difference = tf.math.abs(tensor2-tensor1)
        return difference

    def show_model_summary(self):
        self.model.summary()
        pass

    def compile_the_model(self):
        optimizer = optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=optimizer, metrics=['accuracy'])
        pass

    def get_the_model(self):
        return self.model

    def save_the_model(self):
        self.model.save(self.save_path)
        pass

    def load_the_model(self):
        return tf.keras.models.load_model(self.save_path, custom_objects={'difference_layer': self.difference_layer})


    def define_Model_architecture(self):
        input_shape = (250, 250, 3) #Un-hardcode this later
        input_left = Input(shape=input_shape)
        input_right = Input(shape=input_shape)

        #left_encoding = self.convolutional_layer(input_left, self.num_of_filters)
        #right_encoding = self.convolutional_layer(input_right, self.num_of_filters)

        left_encoding = self.get_encoding_layer(input_layer= input_left, num_of_layers=5, num_of_filters=self.num_of_filters, kernel_size=3)
        right_encoding = self.get_encoding_layer(input_layer= input_right, num_of_layers=5, num_of_filters=self.num_of_filters, kernel_size=3)

        difference_layer = Lambda(self.difference_layer)([left_encoding, right_encoding])
        logistic_layer = Dense(units = 1, activation=sigmoid)(difference_layer)
        self.model = Model(inputs=[input_left, input_right], outputs = logistic_layer)
        pass

    def convolutional_layer(self, input_layer, num_of_filters, kernel_size):
        conv2d_layer = Conv2D(padding = 'same', kernel_size = (kernel_size, kernel_size), strides=(1,1), filters=num_of_filters)(input_layer)
        activation_layer = ReLU()(conv2d_layer)
        dropout_layer = Dropout(rate=0.1)(activation_layer)
        return activation_layer

    def convolutional_block(self, input_layer, num_of_layers, num_of_filters, kernel_size):
        layer = self.convolutional_layer(input_layer=input_layer, num_of_filters=num_of_filters, kernel_size=kernel_size)

        for i in range(num_of_layers-1):
            layer = self.convolutional_layer(input_layer= layer, num_of_filters=num_of_filters*(math.pow(2, i+1)), kernel_size=kernel_size)
            pass

        return layer

    def get_encoding_layer(self, input_layer, num_of_layers, num_of_filters, kernel_size):


        block0 = self.convolutional_block(input_layer=input_layer, num_of_layers=num_of_layers, num_of_filters=num_of_filters, kernel_size=kernel_size)
        max_pool_layer0 = MaxPool2D(padding='valid',strides=(2,2), pool_size=(2,2))(block0)
        reduction_layer0 = Conv2D(padding='same', kernel_size=(1,1), strides = (1,1), filters = 3)(max_pool_layer0)

        block1 = self.convolutional_block(input_layer=reduction_layer0, num_of_layers=num_of_layers, num_of_filters=num_of_filters, kernel_size=kernel_size)
        max_pool_layer1 = MaxPool2D(padding='valid',strides=(2,2), pool_size=(2,2))(block1)
        reduction_layer1 = Conv2D(padding='same', kernel_size=(1, 1), strides=(1, 1), filters=3)(max_pool_layer1)

        block2 = self.convolutional_block(input_layer=reduction_layer1, num_of_layers=num_of_layers, num_of_filters=num_of_filters, kernel_size=kernel_size)
        max_pool_layer2 = MaxPool2D(padding='valid',strides=(2,2), pool_size=(2,2))(block2)
        reduction_layer2 = Conv2D(padding='same', kernel_size=(1, 1), strides=(1, 1), filters=3)(max_pool_layer2)

        block3 = self.convolutional_block(input_layer=reduction_layer2, num_of_layers=num_of_layers, num_of_filters=num_of_filters,kernel_size=kernel_size)
        max_pool_layer3 = MaxPool2D(padding='valid', strides=(2, 2), pool_size=(2, 2))(block3)
        reduction_layer3 = Conv2D(padding='same', kernel_size=(1, 1), strides=(1, 1), filters=3)(max_pool_layer3)

        block4 = self.convolutional_block(input_layer=reduction_layer3, num_of_layers=num_of_layers, num_of_filters=num_of_filters,kernel_size=kernel_size)
        max_pool_layer4 = MaxPool2D(padding='valid', strides=(2, 2), pool_size=(2, 2))(block4)
        reduction_layer4 = Conv2D(padding='same', kernel_size=(1, 1), strides=(1, 1), filters=3)(max_pool_layer4)

        flatten_layer = Flatten()(reduction_layer4)
        dense_layer = Dense(units=100)(flatten_layer)
        dropout_layer = Dropout(rate = 0.1)(dense_layer)
        return dropout_layer
    pass
