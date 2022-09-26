import tensorflow as tf
from keras import Model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Lambda, ReLU, Dropout
from keras.activations import sigmoid
from keras.optimizers import RMSprop
from keras import backend as K



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

    def euclidean_distance(self, vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    def contrastive_loss_with_margin(self, margin):
        def contrastive_loss(y_true, y_pred):
            square_pred = K.square(y_pred)
            margin_square = K.square(K.maximum(margin - y_pred, 0))
            return (y_true * square_pred + (1 - y_true) * margin_square)
        return contrastive_loss

    def define_Model_architecture(self):
        common_block = self.get_encoding_layer(input_shape=self.input_shape, num_of_filters=self.num_of_filters)

        left_input = Input(shape=self.input_shape, name="left_input")
        output_left = common_block(left_input)
        right_input = Input(shape=self.input_shape, name="right_input")
        output_right =common_block(right_input)

        output_layer = Lambda(self.euclidean_distance, name="output_layer")([output_left, output_right])

        self.model = Model(inputs=[left_input, right_input], outputs = output_layer)
        pass

    def get_encoding_layer(self, input_shape, num_of_filters):
        input_layer = Input(shape = input_shape)
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
        dense_layer0 = Dense(units=512, activation = 'relu')(flatten_layer)
        dropout_dense0 = Dropout(rate=0.1)(dense_layer0)
        dense_layer1 = Dense(units=512, activation = 'relu')(dropout_dense0)
        #dropout_dense1 = Dropout(rate=0.1)(dense_layer1)

        #encoding_layer = Dense(units=256, activation='sigmoid')(dropout_dense1)

        model = Model(inputs = input_layer, outputs=dense_layer1)
        return model



    def show_model_summary(self):
        self.model.summary()
        pass

    def compile_the_model(self):
        optimizer1 = RMSprop(learning_rate=0.00001)
        self.model.compile(loss=self.contrastive_loss_with_margin(1), optimizer=optimizer1)
        pass

    def get_the_model(self):
        return self.model

    def save_the_model(self):
        self.model.save(self.save_path)
        pass

    def load_the_model(self):
        return tf.keras.models.load_model(self.save_path, custom_objects={'euclidean_distance': self.euclidean_distance, 'contrastive_loss' : self.contrastive_loss_with_margin})

    pass