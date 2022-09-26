import tensorflow as tf
from keras import Model
from keras.layers import Dense, Input, Lambda, GlobalAveragePooling2D, Dropout, ReLU
from keras.activations import sigmoid
from keras.optimizers import RMSprop
from keras.applications.resnet import ResNet50
from tensorflow import keras
from keras.models import Model
from keras import backend as K


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
        input_shape = self.input_shape
        input_left = Input(shape=input_shape)
        input_right = Input(shape=input_shape)

        base_model = self.get_base_model()


        left_encoding = base_model(input_left)
        right_encoding = base_model(input_right)

        output_layer = Lambda(self.euclidean_distance, name="output_layer")([left_encoding, right_encoding])

        self.model = Model(inputs=[input_left, input_right], outputs = output_layer)
        pass

    def get_base_model(self):

        resnet_50 = ResNet50(include_top=False, weights="imagenet")
        layers = resnet_50.layers
        my_resnet_50 = keras.Model(inputs=resnet_50.input,outputs=resnet_50.get_layer(layers[-1].name).output) #Avoid 2 layers witht the same name
        my_resnet_50.trainable = False
        encoding = my_resnet_50.output
        encoding = GlobalAveragePooling2D()(encoding)
        encoding_layer = Dense(units = 4096, activation = 'relu')(encoding)
        encoding_layer = Dropout(rate = 0.2)(encoding_layer)
        encoding_layer = Dense(units=512, activation = 'relu')(encoding_layer)
        encoding_layer = Dropout(rate=0.2)(encoding_layer)
        encoding_layer = Dense(units=512, activation='relu')(encoding_layer)
        model = Model(inputs = resnet_50.input, outputs = encoding_layer)
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
        return tf.keras.models.load_model(self.save_path, custom_objects={'difference_layer': self.difference_layer})

    pass