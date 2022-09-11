import BuildInputLists as bil
import Siamese_Network_builder_v1 as sn
import Prebuilt_Network as pn
from matplotlib import pyplot as plt
import Siamese_Network_builder_v2 as sn2
import SIamese_Network_builder_v3 as sn3
import Siamese_Network_builder_v4 as sn4
import Resnet50_siamese_network as r50sn
import MobileNet_v3_siamese_network as mv3ssn
import tensorflow as tf


def display_image_from_array(array):
    plt.imshow(array, interpolation='nearest')
    plt.show()

load_model = False

train_size = 4500
val_size = 200
num_of_filters = 32
basePath = "C:/Users/allan/OneDrive/Desktop/Face verification dataset/Face Data/Face Dataset"
save_path = "saved_model/my_model" + str(num_of_filters) + ".h5"
resnet_input_shape = (224, 224, 3)

#siamese_network = sn.Siamese_Network_builder_v1(num_of_filters, save_path)
#siamese_network.define_Model_architecture()
#siamese_network.show_model_summary()
#siamese_network.compile_the_model()

#siamese_network = sn2.Siamese_Network_builder_v2(num_of_filters=num_of_filters, save_path=save_path)
#siamese_network.define_Model_architecture()
#siamese_network.show_model_summary()
#siamese_network.compile_the_model()

# 3x3 conv kernel size
#512->256->128: peak loss= 0.5992, peak accuracy = 0.73
#512->512->256: peak loss = 0.5678, peak accuracy = 0.76 (Potential with more epochs)
#1024->512->256: Fail
#2048: peak loss = 0.5747, peak accuracy = 0.7650

#siamese_network = sn3.Siamese_Network_builder_v3(num_of_filters=num_of_filters, save_path=save_path, input_shape=resnet_input_shape)
#siamese_network.define_Model_architecture()
#siamese_network.show_model_summary()
#siamese_network.compile_the_model()

#siamese_network = sn4.Siamese_Network_builder_v4(num_of_filters=num_of_filters, save_path=save_path)
#siamese_network.define_Model_architecture()
#siamese_network.show_model_summary()
#siamese_network.compile_the_model()

siamese_network = r50sn.Resnet50_siamese_network(save_path=save_path, input_shape=resnet_input_shape)
siamese_network.define_Model_architecture()
siamese_network.compile_the_model()
siamese_network.show_model_summary()

#siamese_network = mv3ssn.Mobilenet_V3_small(save_path=save_path, input_shape=resnet_input_shape)
#siamese_network.define_Model_architecture()
#siamese_network.compile_the_model()
#siamese_network.show_model_summary()

if not load_model:
    model = siamese_network.get_the_model()
else:
    model = siamese_network.load_the_model()


input_list_builder = bil.BuildInputLists(basePath, input_shape=resnet_input_shape)
input_list_builder.build_list_Image_folders()
input_list_builder.build_positive_list()
input_list_builder.build_negative_list(train_size + val_size)  # Size of negative list
train_left_array, train_right_array, train_label_array, validation_left_array, validation_right_array, validation_label_array = input_list_builder.build_input_and_label_matrices(train_size, val_size)

print(train_left_array.shape)
print(validation_left_array.shape)

pretrained_model = pn.Prebuilt_Network(model, train_left_array, train_right_array, train_label_array, validation_left_array, validation_right_array, validation_label_array, save_path)
pretrained_model.train_the_model()
pretrained_model.save_the_model()











