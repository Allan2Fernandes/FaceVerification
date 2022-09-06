import BuildInputLists as bil
import Siamese_Network_builder_v1 as sn
import Prebuilt_Network as pn
from matplotlib import pyplot as plt
import Siamese_Network_builder_v2 as sn2
import SIamese_Network_builder_v3 as sn3
import tensorflow as tf


def display_image_from_array(array):
    plt.imshow(array, interpolation='nearest')
    plt.show()

load_model = False

train_size = 4500
val_size = 200
num_of_filters = 4
basePath = "C:/Users/allan/OneDrive/Desktop/Face verification dataset/Face Data/Face Dataset"
save_path = "saved_model/my_model" + str(num_of_filters) + ".h5"

#siamese_network = sn.Siamese_Network_builder_v1(num_of_filters, save_path)
#siamese_network.define_Model_architecture()
#siamese_network.show_model_summary()
#siamese_network.compile_the_model()

#siamese_network = sn2.Siamese_Network_builder_v2(num_of_filters=num_of_filters, save_path=save_path)
#siamese_network.define_Model_architecture()
#siamese_network.show_model_summary()
#siamese_network.compile_the_model()


siamese_network = sn3.Siamese_Network_builder_v3(32, save_path)
siamese_network.define_Model_architecture()
siamese_network.show_model_summary()
siamese_network.compile_the_model()

if not load_model:
    model = siamese_network.get_the_model()
else:
    model = siamese_network.load_the_model()


#input_list_builder = bil.BuildInputLists(basePath)
#input_list_builder.build_list_Image_folders()
#input_list_builder.build_positive_list()
#input_list_builder.build_negative_list(train_size + val_size)  # Size of negative list
#train_left_array, train_right_array, train_label_array, validation_left_array, validation_right_array, validation_label_array = input_list_builder.build_input_and_label_matrices(train_size, val_size)

#pretrained_model = pn.Prebuilt_Network(model, train_left_array, train_right_array, train_label_array, validation_left_array, validation_right_array, validation_label_array, save_path)
#pretrained_model.train_the_model()
#pretrained_model.save_the_model()











