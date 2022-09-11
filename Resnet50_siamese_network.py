import tensorflow as tf
from keras import Model
from keras.layers import Dense, Input, Lambda, GlobalAveragePooling2D, Flatten, Conv2D, MaxPool2D
from keras.activations import sigmoid
from keras import optimizers
from keras.applications.resnet import ResNet50
from tensorflow import keras


class Resnet50_siamese_network:
    def __init__(self, save_path, input_shape):
        self.save_path = save_path
        self.input_shape = input_shape
        pass

    def difference_layer(self, tensor_set):
        tensor1 = tensor_set[0]
        tensor2 = tensor_set[1]
        difference = tf.math.abs(tensor2-tensor1)
        return difference

    def define_Model_architecture(self):
        input_shape = self.input_shape
        input_left = Input(shape=input_shape)
        input_right = Input(shape=input_shape)
        left_encoding = self.get_encoding_layer(input_left, "left")
        right_encoding = self.get_encoding_layer(input_right, "right")

        difference_layer = Lambda(self.difference_layer)([left_encoding, right_encoding])
        logistic_layer = Dense(units = 1, activation=sigmoid)(difference_layer)
        self.model = Model(inputs=[input_left, input_right], outputs = logistic_layer)
        pass

    def get_encoding_layer(self, input_layer, side):
        resnet_50 = ResNet50(include_top=False, weights="imagenet")
        my_resnet_50 = keras.Model(inputs=resnet_50.input,outputs=resnet_50.get_layer('conv5_block3_out').output,name='resnet50_' + side) #Avoid 2 layers witht the same name
        my_resnet_50.trainable = True
        encoding = my_resnet_50(input_layer)
        encoding = GlobalAveragePooling2D()(encoding)
        encoding = keras.layers.Dense(1000)(encoding)
        encoding_layer = Dense(1000)(encoding)
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