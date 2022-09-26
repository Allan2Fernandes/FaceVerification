import tensorflow as tf
from keras import optimizers
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Lambda, Dropout
from keras import Model


class Siamese_Network_builder_v4:
    def __init__(self,save_path, num_of_filters):
        self.save_path = save_path
        self.num_of_filters = num_of_filters
        pass

    def difference_layer(self, tensor_set):
        tensor1 = tensor_set[0]
        tensor2 = tensor_set[1]
        difference = tf.math.abs(tensor2-tensor1)
        return difference

    def define_Model_architecture(self):
        input_shape = (250, 250, 3) #Un-hardcode this later
        input_left = Input(shape=input_shape)
        input_right = Input(shape=input_shape)
        left_encoding = self.get_encoding_layer(input_left)
        right_encoding = self.get_encoding_layer(input_right)

        difference_layer = Lambda(self.difference_layer)([left_encoding, right_encoding])
        logistic_layer = Dense(units = 1, activation='sigmoid')(difference_layer)
        self.model = Model(inputs=[input_left, input_right], outputs = logistic_layer)
        pass

    def get_encoding_layer(self, input_layer):
        layer0 = Conv2D(filters=self.num_of_filters, kernel_size=(10,10), activation='relu')(input_layer)
        layer1 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(layer0)

        layer2 = Conv2D(filters=self.num_of_filters*2, kernel_size=(7,7), activation='relu')(layer1)
        layer3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(layer2)

        layer4 = Conv2D(filters=self.num_of_filters*2, kernel_size=(4, 4), activation='relu')(layer3)
        layer5 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(layer4)

        layer6 = Conv2D(filters=self.num_of_filters*4, kernel_size=(4, 4), activation='relu')(layer5)

        layer7 = Flatten()(layer6)
        encoding_layer = Dense(units=128)(layer7)
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
    pass