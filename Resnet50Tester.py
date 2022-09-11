
from tensorflow import keras
from keras.applications.resnet import ResNet50
from keras.layers import Input, Dense, Lambda
import tensorflow as tf
from keras.activations import sigmoid

def difference_layer(tensor_set):
    tensor1 = tensor_set[0]
    tensor2 = tensor_set[1]
    difference = tf.math.abs(tensor2 - tensor1)
    return difference

resnet_50_left = ResNet50(weights='imagenet', include_top=False)
resnet_50_left.summary()
#my_resnet_50_left = keras.Model(inputs=resnet_50_left.input, outputs=resnet_50_left.get_layer('conv5_block3_out').output, name='resnet50_left')
#my_resnet_50_left.trainable = False

"""
resnet_50_right = ResNet50(weights='imagenet', include_top=False)
my_resnet_50_right = keras.Model(inputs=resnet_50_right.input, outputs=resnet_50_right.get_layer('conv5_block3_out').output, name='resnet50_right')
my_resnet_50_right.trainable = False

input_left = keras.Input(shape=(224,224,3))
left_encoding = my_resnet_50_left(input_left)
left_encoding = keras.layers.GlobalAveragePooling2D()(left_encoding)
output_left = keras.layers.Dense(512, activation="sigmoid")(left_encoding)

input_right = keras.Input(shape=(224,224,3))
right_encoding = my_resnet_50_right(input_right)
right_encoding = keras.layers.GlobalAveragePooling2D()(right_encoding)
output_right = keras.layers.Dense(512, activation="sigmoid")(right_encoding)

difference_layer = Lambda(difference_layer)([output_left, output_right])
logistic_layer = Dense(units = 1, activation=sigmoid)(difference_layer)



model = keras.Model(inputs=[input_left, input_right], outputs=logistic_layer, name="my_model")
model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()




"""






