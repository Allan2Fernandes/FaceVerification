import tensorflow as tf
from keras.applications.mobilenet_v3 import MobileNetV3Small

model = MobileNetV3Small(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.summary()
print(model.input)
print(model.get_layer("multiply_17").output)