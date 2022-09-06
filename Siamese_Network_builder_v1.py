import tensorflow as tf
from keras import Model
from keras.layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Input, Lambda, BatchNormalization, Dropout
from keras.activations import sigmoid
from keras import optimizers



class Siamese_Network_builder_v1:
    def __init__(self, num_of_filters, save_path):
        self.num_of_filters = num_of_filters
        self.save_path = save_path
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

        left_encoding = self.convolutional_block(input_left, self.num_of_filters)
        right_encoding = self.convolutional_block(input_right, self.num_of_filters)

        difference_layer = Lambda(self.difference_layer)([left_encoding, right_encoding])
        logistic_layer = Dense(units = 1, activation=sigmoid)(difference_layer)
        self.model = Model(inputs=[input_left, input_right], outputs = logistic_layer)
        pass

    def convolutional_block(self, input_layer, starting_num_of_filters):
        conv2dLayer1 = Conv2D(padding = 'same', kernel_size=(3,3), strides=(1,1), filters=starting_num_of_filters)(input_layer)
        normalizationLayer = BatchNormalization()(conv2dLayer1)
        reluLayer1 = ReLU()(normalizationLayer)
        maxPoolLayer1 = MaxPool2D(padding='valid',strides=(2,2), pool_size=(2,2))(reluLayer1)

        conv2dLayer2 = Conv2D(padding='same', kernel_size=(3, 3), strides=(1, 1), filters=starting_num_of_filters*2)(maxPoolLayer1)
        reluLayer2 = ReLU()(conv2dLayer2)
        maxPoolLayer2 = MaxPool2D(padding='valid', strides=(2, 2), pool_size=(2, 2))(reluLayer2)

        conv2dLayer3 = Conv2D(padding='same', kernel_size=(3, 3), strides=(1, 1), filters=starting_num_of_filters*4)(maxPoolLayer2)
        reluLayer3 = ReLU()(conv2dLayer3)
        maxPoolLayer3 = MaxPool2D(padding='valid', strides=(2, 2), pool_size=(2, 2))(reluLayer3)

        conv2dLayer4 = Conv2D(padding='same', kernel_size=(3, 3), strides=(1, 1), filters=starting_num_of_filters*8)(maxPoolLayer3)
        reluLayer4 = ReLU()(conv2dLayer4)
        maxPoolLayer4 = MaxPool2D(padding='valid', strides=(2, 2), pool_size=(2, 2))(reluLayer4)

        conv2dLayer5 = Conv2D(padding='same', kernel_size=(3, 3), strides=(1, 1), filters=starting_num_of_filters * 16)(maxPoolLayer4)
        reluLayer5 = ReLU()(conv2dLayer5)
        maxPoolLayer5 = MaxPool2D(padding='valid', strides=(2, 2), pool_size=(2, 2))(reluLayer5)

        flattenLayer = Flatten()(maxPoolLayer5)
        DenseLayer = Dense(units=100)(flattenLayer) #Num of encoding features
        dropout_layer = Dropout(rate=0.2)(DenseLayer)
        return dropout_layer

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
