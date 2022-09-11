import tensorflow as tf
from keras import Model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Lambda, Dropout
from keras.activations import sigmoid
from keras import optimizers



class Siamese_Network_builder_v3:
    def __init__(self, num_of_filters, save_path, input_shape):
        self.num_of_filters = num_of_filters
        self.save_path = save_path
        self.input_shape = input_shape
        pass

    def difference_layer(self, tensor_set):
        tensor1 = tensor_set[0]
        tensor2 = tensor_set[1]
        difference = tf.math.abs(tensor2-tensor1)
        return difference

    def define_Model_architecture(self):
        input_shape = self.input_shape #Un-hardcode this later
        input_left = Input(shape=input_shape)
        input_right = Input(shape=input_shape)
        left_encoding = self.get_encoding_layer(input_left, self.num_of_filters)
        right_encoding = self.get_encoding_layer(input_right, self.num_of_filters)

        difference_layer = Lambda(self.difference_layer)([left_encoding, right_encoding])
        logistic_layer = Dense(units = 1, activation=sigmoid)(difference_layer)
        self.model = Model(inputs=[input_left, input_right], outputs = logistic_layer)
        pass

    def get_encoding_layer(self, input_layer, num_of_filters):
        conv_layer0 = Conv2D(padding = 'same', kernel_size=(3,3), strides = (1,1), filters=num_of_filters, activation='relu')(input_layer)
        max_pooling0 = MaxPool2D(padding = 'same', strides = (2,2), pool_size=(2,2))(conv_layer0)

        conv_layer1 = Conv2D(padding = 'same', kernel_size=(3,3), strides = (1,1), filters = num_of_filters*2, activation='relu')(max_pooling0)
        max_pooling1 = MaxPool2D(padding = 'same', strides = (2,2), pool_size=(2,2))(conv_layer1)

        conv_layer2 = Conv2D(padding = 'same', kernel_size=(3,3), strides=(1,1), filters = num_of_filters*4, activation='relu')(max_pooling1)
        max_pooling2 = MaxPool2D(padding='same', strides=(2, 2), pool_size=(2, 2))(conv_layer2)

        conv_layer3 = Conv2D(padding='same', kernel_size=(3,3), strides=(1,1), filters = num_of_filters*8, activation='relu')(max_pooling2)
        conv_layer4 = Conv2D(padding = 'same', kernel_size=(3,3), strides=(1,1), filters = num_of_filters*8, activation='relu')(conv_layer3)
        max_pooling3 = MaxPool2D(padding='same', strides=(2, 2), pool_size=(2, 2))(conv_layer4)

        conv_layer5 = Conv2D(padding='same', kernel_size=(3, 3), strides=(1, 1), filters=num_of_filters*8, activation='relu')(max_pooling3)
        conv_layer6 = Conv2D(padding='same', kernel_size=(3, 3), strides=(1, 1), filters=num_of_filters*8, activation='relu')(conv_layer5)
        max_pooling4 = MaxPool2D(padding='same', strides=(2, 2), pool_size=(2, 2))(conv_layer6)

        flatten_layer = Flatten()(max_pooling4)
        dense_layer0 = Dense(units=512, activation='relu')(flatten_layer)
        dropout_dense0 = Dropout(rate=0.1)(dense_layer0)
        dense_layer1 = Dense(units=512, activation='relu')(dropout_dense0)
        dropout_dense1 = Dropout(rate=0.1)(dense_layer1)

        encoding_layer = Dense(units=256, activation='relu')(dropout_dense1)
        return encoding_layer



    def show_model_summary(self):
        self.model.summary()
        pass

    def compile_the_model(self):
        optimizer = optimizers.Adam(learning_rate = 0.00001, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-07)
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=optimizer, metrics=['accuracy'])
        pass

    def get_the_model(self):
        return self.model

    def save_the_model(self):
        self.model.save(self.save_path)
        pass

    def load_the_model(self):
        return tf.keras.models.load_model(self.save_path, custom_objects={'difference_layer': self.difference_layer})

    pass