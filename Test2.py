from keras.applications.resnet import ResNet50
from keras import Model

resnet_50 = ResNet50(weights='imagenet', include_top=False)
resnet_50.summary()

my_resnet_50 = Model(inputs=resnet_50.input, outputs=resnet_50.get_layer('conv5_block3_out').output, name='resnet50_1')

my_resnet_50.summary()